//! Booster：训练产物 + `fit` GBDT 循环（对 L2 / binary logloss 通用）。
//!
//! 算法（m0-spec §4）：
//! 1. 分箱：`BinTable::build_from_dataset`（排序精确分位，无 seed 也确定）；
//! 2. 初始化：`loss.init_score(y)`；
//! 3. 每轮：由当前预测算梯度/海森 → `TreeBuilder` 建树 → `pred += lr·tree`；
//! 4. 预测：`raw = init + Σ lr·tree`，`final = loss.transform(raw)`。
//!
//! 预测不依赖 BinTable：树存真实阈值，任意同特征数 Dataset 可直接推断
//! （类别特征经编码解析后同样走数值树）。

use arrow::array::{Array, Float64Array};
use std::borrow::Cow;

use crate::binning::BinTable;
use crate::data::missing::is_missing_value;
use crate::data::target_stats::{
    CategoricalEncoding, apply_encoding, compute_ordered_ts, resolve_to_dataset,
};
use crate::data::{DataError, Dataset};
use crate::loss::Loss;
use crate::tree::{Tree, TreeBuilder};

use super::context::TrainingContext;
use super::error::BoostingError;
use super::params::BoostingParams;

/// 训练完成的梯度提升模型。
///
/// 自包含（含 BinTable + 类别编码）：预测只用树（真实阈值），但模型序列化契约
/// （contracts §1.2）要求模型自包含，故训练后保留分箱表供 save/热替换。
#[derive(Debug)]
pub struct Booster<L: Loss> {
    loss: L,
    trees: Vec<Tree>,
    table: BinTable,
    init_score: f64,
    learning_rate: f64,
    /// 类别特征编码（M1-4，D9；无类别特征时为 None）。
    encoding: Option<CategoricalEncoding>,
    /// 原始特征索引中哪些是类别特征（与 encoding 对齐）。
    cat_features: Vec<usize>,
}

impl<L: Loss> Booster<L> {
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    pub fn init_score(&self) -> f64 {
        self.init_score
    }

    pub fn loss(&self) -> &L {
        &self.loss
    }

    /// 模型自包含的分箱表（训练集派生的分箱边界）。
    pub fn bin_table(&self) -> &BinTable {
        &self.table
    }

    /// 类别特征编码（有类别特征时 Some）。
    pub fn categorical_encoding(&self) -> Option<&CategoricalEncoding> {
        self.encoding.as_ref()
    }

    /// 单棵树序列化访问器（model/ 模块用）。
    pub(crate) fn trees(&self) -> &[Tree] {
        &self.trees
    }

    pub(crate) fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub(crate) fn cat_features(&self) -> &[usize] {
        &self.cat_features
    }

    /// model/ 反序列化重建（结构已校验）。
    pub(crate) fn from_parts(
        loss: L,
        trees: Vec<Tree>,
        table: BinTable,
        init_score: f64,
        learning_rate: f64,
        encoding: Option<CategoricalEncoding>,
        cat_features: Vec<usize>,
    ) -> Self {
        Self {
            loss,
            trees,
            table,
            init_score,
            learning_rate,
            encoding,
            cat_features,
        }
    }

    /// 序列化为字节（contracts §1.2 显式布局 + checksum）。
    pub fn serialize(&self) -> Vec<u8> {
        crate::model::io::serialize(self)
    }

    /// 从字节反序列化；`loss` 用于校验模型头损失名并重建。
    pub fn deserialize(bytes: &[u8], loss: L) -> Result<Self, crate::model::ModelError> {
        crate::model::io::deserialize(bytes, loss)
    }

    /// 解析推断集：类别特征经编码转数值（OOV → 先验）；无类别则原样借用。
    fn resolve<'a>(&self, ds: &'a Dataset) -> Result<Cow<'a, Dataset>, BoostingError> {
        if let Some(enc) = &self.encoding {
            let resolved = apply_encoding(ds, enc, &self.cat_features)?;
            let rd = resolve_to_dataset(ds, &resolved)?;
            Ok(Cow::Owned(rd))
        } else {
            Ok(Cow::Borrowed(ds))
        }
    }

    /// 任意数据集的原始分数（init + Σ lr·tree），未过 transform。
    pub fn raw_scores(&self, ds: &Dataset) -> Result<Vec<f64>, BoostingError> {
        let resolved = self.resolve(ds)?;
        let ds = resolved.as_ref();
        let n = ds.num_rows();
        let cols = feature_columns(ds)?;
        let policy = ds.missing_policy();
        let mut out = vec![self.init_score; n];
        for tree in &self.trees {
            for (r, o) in out.iter_mut().enumerate() {
                *o += self.learning_rate * predict_row(tree, &cols, r, policy);
            }
        }
        Ok(out)
    }

    /// 最终预测（transform 后）：L2 → 原值；binary logloss → 概率。
    pub fn predict(&self, ds: &Dataset) -> Result<Vec<f64>, BoostingError> {
        Ok(self
            .raw_scores(ds)?
            .into_iter()
            .map(|raw| self.loss.transform(raw))
            .collect())
    }

    /// 单行最终预测（数组输入：`values[f]` / `is_missing[f]`）。
    pub fn predict_row(&self, values: &[f64], is_missing: &[bool]) -> f64 {
        let mut raw = self.init_score;
        for tree in &self.trees {
            raw += self.learning_rate * tree.predict(values, is_missing);
        }
        self.loss.transform(raw)
    }
}

/// 拟合：分箱 → 迭代提升 → 返回训练产物。
///
/// `ctx` 为确定性契约的显式载体（红线 4/红线 3）：seed 用于 ordered TS 的
/// permutation 派生（防泄漏，D9）；同输入同 seed 恒逐位一致。
pub fn fit<L: Loss>(
    ds: &Dataset,
    params: &BoostingParams,
    loss: L,
    ctx: &TrainingContext,
) -> Result<Booster<L>, BoostingError> {
    // 类别特征 → ordered TS → 数值化解析数据集（M1-4，D9）
    let cat_features: Vec<usize> = (0..ds.num_features())
        .filter(|&f| ds.feature_is_categorical(f))
        .collect();
    let (encoding, resolved_ds): (Option<CategoricalEncoding>, Cow<Dataset>) = if cat_features
        .is_empty()
    {
        (None, Cow::Borrowed(ds))
    } else {
        for &f in &cat_features {
            if let Some(len) = ds.categorical_dictionary_len(f)?
                && len > params.max_categories
            {
                return Err(BoostingError::Data(DataError::TooManyCategories {
                    name: ds.feature_names()[f].clone(),
                    got: len,
                    limit: params.max_categories,
                }));
            }
        }
        let (resolved, enc) = compute_ordered_ts(ds, &cat_features, params.categorical_alpha, ctx)?;
        let rd = resolve_to_dataset(ds, &resolved)?;
        (Some(enc), Cow::Owned(rd))
    };

    let ds = resolved_ds.as_ref();
    let n = ds.num_rows();
    let y: Vec<f64> = ds.target_values().values().to_vec();
    let (table, matrix) = BinTable::build_from_dataset(ds, params.max_bins)?;
    let cols = feature_columns(ds)?;
    let policy = ds.missing_policy();

    let init_score = loss.init_score(&y);
    let mut pred = vec![init_score; n];
    let mut trees = Vec::with_capacity(params.n_estimators);
    let builder = TreeBuilder::new(params.tree_params);

    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n];
    for _ in 0..params.n_estimators {
        for i in 0..n {
            grad[i] = loss.gradient(y[i], pred[i]);
            hess[i] = loss.hessian(y[i], pred[i]);
        }
        let tree = builder.build(&matrix, &table, &grad, &hess)?;
        for (r, p) in pred.iter_mut().enumerate() {
            *p += params.learning_rate * predict_row(&tree, &cols, r, policy);
        }
        trees.push(tree);
    }

    Ok(Booster {
        loss,
        trees,
        table,
        init_score,
        learning_rate: params.learning_rate,
        encoding,
        cat_features,
    })
}

/// 单棵树单行推断（零拷贝借用特征列视图）。
fn predict_row(
    tree: &Tree,
    cols: &[&Float64Array],
    row: usize,
    policy: crate::data::MissingPolicy,
) -> f64 {
    tree.predict_one(|f| {
        let col = cols[f];
        let v = col.value(row);
        (v, is_missing_value(v, col.is_null(row), policy))
    })
}

/// 预先取出全部特征列视图（构造后取列不再失败，此处集中处理 Result）。
fn feature_columns(ds: &Dataset) -> Result<Vec<&Float64Array>, BoostingError> {
    let mut cols = Vec::with_capacity(ds.num_features());
    for f in 0..ds.num_features() {
        cols.push(ds.feature_values(f)?);
    }
    Ok(cols)
}
