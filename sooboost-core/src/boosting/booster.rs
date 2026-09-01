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

/// 特征重要度口径（M6-3）。
///
/// - `Gain`：该特征各次分裂的增益之和（最常用，XGBoost 同名口径）；
/// - `Cover`：该特征各次分裂节点的覆盖样本数之和；
/// - `Frequency`：该特征被用作分裂的次数。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImportanceKind {
    /// 分裂增益之和。
    #[default]
    Gain,
    /// 覆盖样本数之和。
    Cover,
    /// 分裂次数。
    Frequency,
}

/// 早停配置（M6-1）。
///
/// 每轮提升后在 `eval_set` 上评估损失值（`Loss::value` 均值），
/// 连续 `rounds` 轮无改善则停止，并把树集合回滚到最优轮
/// （`Booster::num_trees()` 即最优轮树数，序列化契约不变）。
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// 验证集（拥有数据；类别特征将用训练集学到的编码解析）。
    pub eval_set: Dataset,
    /// patience：验证损失连续 `rounds` 轮无改善则停（必须 ≥ 1）。
    pub rounds: usize,
}

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
    /// 实际使用的提升轮数（无早停 = n_estimators；早停 = 最优轮 + 1）。
    best_iteration: usize,
    /// 验证集损失历史（每轮一个值；无早停训练为空）。
    eval_history: Vec<f64>,
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

    /// 实际使用的提升轮数（早停回滚后可能小于请求的 n_estimators）。
    pub fn best_iteration(&self) -> usize {
        self.best_iteration
    }

    /// 验证集损失历史（每轮一个值；未启用早停则为空切片）。
    pub fn eval_history(&self) -> &[f64] {
        &self.eval_history
    }

    /// 特征重要度（归一化到和为 1；M6-3）。
    ///
    /// 全部特征都未参与分裂（极端退化）时返回全 0 向量。
    /// 数据源为树节点持久化的 gain/cover（v3 格式），因此 load 后同样可用。
    #[must_use]
    pub fn feature_importances(&self, kind: ImportanceKind) -> Vec<f64> {
        let nf = self.table.num_features();
        let mut acc = vec![0.0f64; nf];
        for tree in &self.trees {
            for i in 0..tree.num_nodes() {
                if tree.left()[i] >= 0 {
                    let f = tree.split_features()[i];
                    acc[f] += match kind {
                        ImportanceKind::Gain => tree.split_gains()[i],
                        ImportanceKind::Cover => tree.node_counts()[i],
                        ImportanceKind::Frequency => 1.0,
                    };
                }
            }
        }
        let total: f64 = acc.iter().sum();
        if total <= 0.0 {
            return vec![0.0; nf];
        }
        acc.iter().map(|&v| v / total).collect()
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

    /// 训练时的学习率（模型格式持久化字段，contracts §1.2；门面 `api` 回填配置用）。
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    pub(crate) fn cat_features(&self) -> &[usize] {
        &self.cat_features
    }

    /// model/ 反序列化重建（结构已校验）。
    ///
    /// 早停信息（best_iteration/eval_history）不属于模型格式：载入后
    /// best_iteration = 树数，eval_history 为空。
    pub(crate) fn from_parts(
        loss: L,
        trees: Vec<Tree>,
        table: BinTable,
        init_score: f64,
        learning_rate: f64,
        encoding: Option<CategoricalEncoding>,
        cat_features: Vec<usize>,
    ) -> Self {
        let best_iteration = trees.len();
        Self {
            loss,
            trees,
            table,
            init_score,
            learning_rate,
            encoding,
            cat_features,
            best_iteration,
            eval_history: Vec::new(),
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
    fit_impl(ds, params, loss, ctx, None)
}

/// 拟合 + 早停（M6-1）。
///
/// 每轮在 `es.eval_set` 上评估损失均值，连续 `es.rounds` 轮无改善则停，
/// 树集合回滚到最优轮。验证集类别特征用训练集编码解析（含 OOV → 先验）。
pub fn fit_with_early_stopping<L: Loss>(
    ds: &Dataset,
    params: &BoostingParams,
    loss: L,
    ctx: &TrainingContext,
    es: &EarlyStopping,
) -> Result<Booster<L>, BoostingError> {
    fit_impl(ds, params, loss, ctx, Some(es))
}

fn fit_impl<L: Loss>(
    ds: &Dataset,
    params: &BoostingParams,
    loss: L,
    ctx: &TrainingContext,
    early_stopping: Option<&EarlyStopping>,
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

    // 早停验证集：入口校验 + 用训练编码解析（与 Booster::resolve 同一语义）
    let eval: Option<EvalState> = match early_stopping {
        None => None,
        Some(es) => {
            if es.rounds == 0 {
                return Err(BoostingError::InvalidEarlyStopping(
                    "rounds（patience）至少为 1",
                ));
            }
            if es.eval_set.num_rows() == 0 {
                return Err(BoostingError::Data(DataError::EmptyDataset));
            }
            if es.eval_set.num_features() != ds.num_features() {
                return Err(BoostingError::EvalSetFeatureMismatch {
                    train: ds.num_features(),
                    eval: es.eval_set.num_features(),
                });
            }
            let resolved_eval = match &encoding {
                None => Cow::Borrowed(&es.eval_set),
                Some(enc) => {
                    let resolved = apply_encoding(&es.eval_set, enc, &cat_features)?;
                    Cow::Owned(resolve_to_dataset(&es.eval_set, &resolved)?)
                }
            };
            let eval_y: Vec<f64> = resolved_eval.target_values().values().to_vec();
            // 列 clone 为拥有（arrow Arc 共享缓冲，零拷贝），避免自引用借用
            let mut eval_cols_owned = Vec::with_capacity(resolved_eval.num_features());
            for f in 0..resolved_eval.num_features() {
                eval_cols_owned.push(resolved_eval.feature_values(f)?.clone());
            }
            let eval_policy = resolved_eval.missing_policy();
            Some(EvalState {
                y: eval_y,
                cols: eval_cols_owned,
                policy: eval_policy,
            })
        }
    };

    let (table, matrix) = BinTable::build_from_dataset(ds, params.max_bins)?;
    let cols = feature_columns(ds)?;
    let policy = ds.missing_policy();

    let init_score = loss.init_score(&y);
    let mut pred = vec![init_score; n];
    let mut trees = Vec::with_capacity(params.n_estimators);
    let builder = TreeBuilder::new(params.tree_params);

    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n];

    // 早停状态：eval pred 由 init 起步，每轮与训练 pred 同步累加
    let mut eval_pred: Vec<f64> = eval
        .as_ref()
        .map(|e| vec![init_score; e.y.len()])
        .unwrap_or_default();
    let mut eval_history: Vec<f64> = Vec::new();
    let mut best_loss = f64::INFINITY;
    let mut best_round = 0usize;
    let mut rounds_since_best = 0usize;

    for round in 0..params.n_estimators {
        for i in 0..n {
            grad[i] = loss.gradient(y[i], pred[i]);
            hess[i] = loss.hessian(y[i], pred[i]);
        }
        let tree = builder.build(&matrix, &table, &grad, &hess)?;
        for (r, p) in pred.iter_mut().enumerate() {
            *p += params.learning_rate * predict_row(&tree, &cols, r, policy);
        }
        trees.push(tree);

        // 早停评估：损失值 = mean(loss.value(y, transform(raw)))（Loss::value 契约）
        if let Some(es) = early_stopping {
            let state = eval.as_ref().expect("eval 与 early_stopping 同步存在");
            let n_eval = state.y.len();
            let eval_cols: Vec<&Float64Array> = state.cols.iter().collect();
            for (r, p) in eval_pred.iter_mut().enumerate() {
                *p +=
                    params.learning_rate * predict_row(&trees[round], &eval_cols, r, state.policy);
            }
            let mut sum = 0.0;
            for (&yi, &pi) in state.y.iter().zip(eval_pred.iter()) {
                sum += loss.value(yi, loss.transform(pi));
            }
            let eval_loss = sum / n_eval as f64;
            eval_history.push(eval_loss);

            if eval_loss < best_loss {
                best_loss = eval_loss;
                best_round = round;
                rounds_since_best = 0;
            } else {
                rounds_since_best += 1;
                if rounds_since_best >= es.rounds {
                    break;
                }
            }
        }
    }

    // 早停回滚：只保留到最优轮（序列化契约不变——树数本就可变）
    if early_stopping.is_some() && trees.len() > best_round + 1 {
        trees.truncate(best_round + 1);
    }

    let best_iteration = trees.len();
    Ok(Booster {
        loss,
        trees,
        table,
        init_score,
        learning_rate: params.learning_rate,
        encoding,
        cat_features,
        best_iteration,
        eval_history,
    })
}

/// 早停验证集的解析状态（内部；列 clone 为拥有，规避自引用借用）。
struct EvalState {
    y: Vec<f64>,
    cols: Vec<Float64Array>,
    policy: crate::data::MissingPolicy,
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
