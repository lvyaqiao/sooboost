//! 原生 softmax 多分类（D8：M1 多分类一期）。
//!
//! 每轮提升对每个类建一棵树：梯度 g_k = p_k − 1{y=k}，海森用对角线近似
//! h_k = p_k·(1−p_k)（标准 GBDT 多分类）。预测 = 各类原始分数累加 → softmax。
//!
//! 注意：与 `Booster<L>`（标量损失）分离；本类型自带训练与预测。

use arrow::array::{Array, Float64Array};

use crate::binning::BinTable;
use crate::boosting::booster::{EarlyStopping, ImportanceKind};
use crate::data::missing::is_missing_value;
use crate::data::{DataError, Dataset, MissingPolicy};
use crate::tree::{Tree, TreeBuilder, TreeParams};

use super::context::TrainingContext;
use super::error::BoostingError;
use super::params::BoostingParams;

/// 多分类训练产物（每类一棵树序列）。
#[derive(Debug)]
pub struct MulticlassBooster {
    n_classes: usize,
    /// `trees[class][tree_idx]`
    trees: Vec<Vec<Tree>>,
    table: BinTable,
    /// 每类初始 logit（类先验 log）。
    init_scores: Vec<f64>,
    learning_rate: f64,
    /// 实际使用的每类轮数（早停回滚后可能小于请求的 n_estimators；M7-1）。
    best_iteration: usize,
    /// 验证集多分类 logloss 历史（每轮一个值；无早停训练为空；M7-1）。
    eval_history: Vec<f64>,
}

impl MulticlassBooster {
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    pub fn num_trees_per_class(&self) -> usize {
        self.trees.first().map_or(0, |v| v.len())
    }

    /// 总树数（K 类 × 每类轮数）。
    pub fn num_trees(&self) -> usize {
        self.trees.iter().map(Vec::len).sum()
    }

    pub fn init_scores(&self) -> &[f64] {
        &self.init_scores
    }

    /// 学习率（io 序列化与门面配置回填用）。
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// 类主序平铺的全部树（io 序列化用，M6-5a）。
    pub(crate) fn trees_flat(&self) -> impl Iterator<Item = &Tree> {
        self.trees.iter().flatten()
    }

    /// 由各部件重建（反序列化路径；结构合法性由 io 层校验，此处不再重复）。
    pub(crate) fn from_parts(
        n_classes: usize,
        trees: Vec<Vec<Tree>>,
        table: BinTable,
        init_scores: Vec<f64>,
        learning_rate: f64,
    ) -> Self {
        let best_iteration = trees.first().map_or(0, Vec::len);
        Self {
            n_classes,
            trees,
            table,
            init_scores,
            learning_rate,
            best_iteration,
            eval_history: Vec::new(),
        }
    }

    /// 实际使用的每类提升轮数（早停回滚后可能小于请求的 n_estimators；M7-1）。
    pub fn best_iteration(&self) -> usize {
        self.best_iteration
    }

    /// 验证集多分类 logloss 历史（每轮一个值；未启用早停则为空切片；M7-1）。
    pub fn eval_history(&self) -> &[f64] {
        &self.eval_history
    }

    /// 特征重要度（跨全部类别的树聚合后归一化到和为 1；M6-5a）。
    ///
    /// 数据源为树节点持久化的 gain/cover（v4 格式），load 后同样可用；
    /// 极端退化（无任何分裂）时返回全 0。
    #[must_use]
    pub fn feature_importances(&self, kind: ImportanceKind) -> Vec<f64> {
        let nf = self.table.num_features();
        let mut acc = vec![0.0f64; nf];
        for tree in self.trees_flat() {
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

    /// 单行各类 logits（`init + Σ lr·tree`；门面 `predict_row` 用，M6-5a）。
    pub(crate) fn raw_logits_row(&self, values: &[f64], is_missing: &[bool]) -> Vec<f64> {
        let mut out = self.init_scores.clone();
        for (k, trees) in self.trees.iter().enumerate() {
            for tree in trees {
                out[k] += self.learning_rate * tree.predict_one(|f| (values[f], is_missing[f]));
            }
        }
        out
    }

    /// 模型自包含的分箱表（与 Booster 一致，供序列化/热替换）。
    pub fn bin_table(&self) -> &BinTable {
        &self.table
    }

    /// 类别概率矩阵 `probs[row][class]`（softmax）。
    pub fn predict_proba(&self, ds: &Dataset) -> Result<Vec<Vec<f64>>, BoostingError> {
        let raw = self.raw_logits(ds)?;
        Ok(raw.iter().map(|row| softmax(row)).collect())
    }

    /// 预测类别（argmax；并列取小类）。
    pub fn predict(&self, ds: &Dataset) -> Result<Vec<usize>, BoostingError> {
        let proba = self.predict_proba(ds)?;
        Ok(proba
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect())
    }

    /// 序列化为字节（v4 多分类布局；M6-5a）。
    pub fn serialize(&self) -> Vec<u8> {
        crate::model::io::serialize_multiclass(self)
    }

    /// 从字节反序列化；损失名必须为 `multiclass_softmax`。
    pub fn deserialize(bytes: &[u8]) -> Result<Self, crate::model::ModelError> {
        crate::model::io::deserialize_multiclass(bytes)
    }

    /// 每行原始 logits（init + Σ lr·tree）。
    pub fn raw_logits(&self, ds: &Dataset) -> Result<Vec<Vec<f64>>, BoostingError> {
        let n = ds.num_rows();
        let cols = feature_columns(ds)?;
        let policy = ds.missing_policy();
        let mut out = vec![self.init_scores.clone(); n];
        for k in 0..self.n_classes {
            for tree in &self.trees[k] {
                for (r, row) in out.iter_mut().enumerate() {
                    row[k] += self.learning_rate * predict_row(tree, &cols, r, policy);
                }
            }
        }
        Ok(out)
    }

    /// 温度缩放校准（M7-2）：在 `ds` 上求最小化 NLL 的温度 T。
    ///
    /// 搜索完全确定：对数均匀粗网格（200 点，T ∈ [0.05, 20]）+ 最优点邻域
    /// 黄金分割细化（80 次迭代）。同数据同模型 → 同 T（红线 3）。
    /// 校准不改变模型本身——T 由调用方持有并传给
    /// [`Self::predict_proba_with_temperature`]（不属于模型格式，避免破坏 v4）。
    pub fn calibrate_temperature(&self, ds: &Dataset) -> Result<f64, BoostingError> {
        if ds.num_rows() == 0 {
            return Err(BoostingError::Data(DataError::EmptyDataset));
        }
        let labels = to_labels(ds.target_values().values(), self.n_classes)?;
        let logits = self.raw_logits(ds)?;
        let n = logits.len();
        let k = self.n_classes;

        let nll = |t: f64| -> f64 {
            let mut row = vec![0.0f64; k];
            let mut sum = 0.0;
            for (r, lrow) in logits.iter().enumerate() {
                for (c, &x) in lrow.iter().enumerate() {
                    row[c] = x / t;
                }
                let probs = softmax(&row);
                sum -= probs[labels[r]].ln();
            }
            sum / n as f64
        };

        // 粗网格（对数均匀）
        const GRID: usize = 200;
        let (t_lo, t_hi) = (0.05f64, 20.0f64);
        let step = (t_hi / t_lo).ln() / (GRID - 1) as f64;
        let mut best_i = 0usize;
        let mut best_v = f64::INFINITY;
        for i in 0..GRID {
            let t = t_lo * (step * i as f64).exp();
            let v = nll(t);
            if v < best_v {
                best_v = v;
                best_i = i;
            }
        }
        // 最优点邻域细化为黄金分割搜索区间
        let lo = if best_i == 0 {
            t_lo
        } else {
            t_lo * (step * (best_i - 1) as f64).exp()
        };
        let hi = if best_i + 1 >= GRID {
            t_hi
        } else {
            t_lo * (step * (best_i + 1) as f64).exp()
        };

        // 黄金分割搜索（单峰区间内最小化）
        const PHI: f64 = 0.618_033_988_749_894_9;
        let mut a = lo;
        let mut b = hi;
        let mut c = b - PHI * (b - a);
        let mut d = a + PHI * (b - a);
        let mut fc = nll(c);
        let mut fd = nll(d);
        for _ in 0..80 {
            if fc < fd {
                b = d;
                d = c;
                fd = fc;
                c = b - PHI * (b - a);
                fc = nll(c);
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + PHI * (b - a);
                fd = nll(d);
            }
        }
        Ok((a + b) / 2.0)
    }

    /// 带温度的类别概率矩阵 `probs[row][class]`（softmax(logits / T)；M7-2）。
    ///
    /// `temperature` 必须为正的有限值（T=1 与 [`Self::predict_proba`] 恒等）。
    pub fn predict_proba_with_temperature(
        &self,
        ds: &Dataset,
        temperature: f64,
    ) -> Result<Vec<Vec<f64>>, BoostingError> {
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(BoostingError::InvalidTemperature(temperature));
        }
        let raw = self.raw_logits(ds)?;
        Ok(raw
            .iter()
            .map(|row| {
                let scaled: Vec<f64> = row.iter().map(|&x| x / temperature).collect();
                softmax(&scaled)
            })
            .collect())
    }
}

/// 拟合多分类模型。`y` 必须为整数标签 ∈ [0, n_classes)。
pub fn fit_multiclass(
    ds: &Dataset,
    params: &BoostingParams,
    n_classes: usize,
    ctx: &TrainingContext,
) -> Result<MulticlassBooster, BoostingError> {
    // ctx（seed 载体）当前仅标量 ordered TS 使用；多分类暂不支持类别特征，
    // 保留参数以与标量 fit 契约对齐，红线 4 语义不变。
    let _ = ctx;
    fit_impl(ds, params, n_classes, None)
}

/// 拟合多分类模型并启用早停（M7-1）。
///
/// 每轮（K 棵类树全部完成后）在 `es.eval_set` 上评估多分类 logloss
/// `-mean ln p[true]`，patience 轮无改善则停，树集合回滚到最优轮。
/// 验证集特征数必须与训练集一致、标签必须合法，否则显式报错。
pub fn fit_multiclass_with_early_stopping(
    ds: &Dataset,
    params: &BoostingParams,
    n_classes: usize,
    ctx: &TrainingContext,
    early_stopping: &EarlyStopping,
) -> Result<MulticlassBooster, BoostingError> {
    let _ = ctx;
    fit_impl(ds, params, n_classes, Some(early_stopping))
}

fn fit_impl(
    ds: &Dataset,
    params: &BoostingParams,
    n_classes: usize,
    early_stopping: Option<&EarlyStopping>,
) -> Result<MulticlassBooster, BoostingError> {
    if n_classes < 2 {
        return Err(BoostingError::Data(DataError::InvalidMulticlassClasses(
            n_classes,
        )));
    }
    let n = ds.num_rows();
    let y: Vec<f64> = ds.target_values().values().to_vec();
    let labels = to_labels(&y, n_classes)?;

    // 早停验证集：入口校验 + 拥有式克隆列（与标量 fit_with_early_stopping 同纪律）
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
            let eval_y: Vec<f64> = es.eval_set.target_values().values().to_vec();
            let eval_labels = to_labels(&eval_y, n_classes)?;
            let mut eval_cols_owned = Vec::with_capacity(es.eval_set.num_features());
            for f in 0..es.eval_set.num_features() {
                eval_cols_owned.push(es.eval_set.feature_values(f)?.clone());
            }
            Some(EvalState {
                labels: eval_labels,
                cols: eval_cols_owned,
                policy: es.eval_set.missing_policy(),
            })
        }
    };

    let (table, matrix) = BinTable::build_from_dataset(ds, params.max_bins)?;
    let cols = feature_columns(ds)?;
    let policy = ds.missing_policy();
    let tree_params: TreeParams = params.tree_params;

    // 类先验 logit
    let mut counts = vec![0usize; n_classes];
    for &l in &labels {
        counts[l] += 1;
    }
    let total = n.max(1) as f64;
    let init_scores: Vec<f64> = counts
        .iter()
        .map(|&c| ((c as f64 / total).clamp(1e-12, 1.0 - 1e-12)).ln())
        .collect();

    // pred[k][i]
    let mut pred: Vec<Vec<f64>> = (0..n_classes).map(|k| vec![init_scores[k]; n]).collect();
    let mut trees: Vec<Vec<Tree>> = (0..n_classes).map(|_| Vec::new()).collect();
    let builder = TreeBuilder::new(tree_params);

    // 早停状态：验证集 logits 由 init 起步，每棵类树完成后累加该类列
    let n_eval = eval.as_ref().map_or(0, |e| e.labels.len());
    let mut eval_logits: Vec<Vec<f64>> = (0..n_classes)
        .map(|k| vec![init_scores[k]; n_eval])
        .collect();
    let mut eval_history: Vec<f64> = Vec::new();
    let mut best_loss = f64::INFINITY;
    let mut best_round = 0usize;
    let mut rounds_since_best = 0usize;

    let mut logits = vec![0.0f64; n_classes];
    let mut grad = vec![0.0f64; n];
    let mut hess = vec![0.0f64; n];
    for round in 0..params.n_estimators {
        for k in 0..n_classes {
            for i in 0..n {
                for (c, p) in logits.iter_mut().enumerate() {
                    *p = pred[c][i];
                }
                let probs = softmax(&logits);
                let is_target = (labels[i] == k) as u8 as f64;
                grad[i] = probs[k] - is_target;
                hess[i] = probs[k] * (1.0 - probs[k]);
            }
            let tree = builder.build(&matrix, &table, &grad, &hess)?;
            for (r, p) in pred[k].iter_mut().enumerate() {
                *p += params.learning_rate * predict_row(&tree, &cols, r, policy);
            }
            if let Some(es) = eval.as_ref() {
                let eval_cols: Vec<&Float64Array> = es.cols.iter().collect();
                for (r, p) in eval_logits[k].iter_mut().enumerate() {
                    *p += params.learning_rate * predict_row(&tree, &eval_cols, r, es.policy);
                }
            }
            trees[k].push(tree);
        }

        // 早停评估：多分类 logloss = -mean ln softmax(logits)[true]
        if let Some(es) = eval.as_ref() {
            let mut sum = 0.0;
            let mut row_logits = vec![0.0f64; n_classes];
            for r in 0..n_eval {
                for (c, col) in eval_logits.iter().enumerate() {
                    row_logits[c] = col[r];
                }
                let probs = softmax(&row_logits);
                sum -= probs[es.labels[r]].ln();
            }
            let eval_loss = sum / n_eval as f64;
            eval_history.push(eval_loss);

            if eval_loss < best_loss {
                best_loss = eval_loss;
                best_round = round;
                rounds_since_best = 0;
            } else {
                rounds_since_best += 1;
                if rounds_since_best
                    >= early_stopping
                        .expect("eval 与 early_stopping 同步存在")
                        .rounds
                {
                    break;
                }
            }
        }
    }

    // 早停回滚：每类只保留到最优轮（与标量 Booster 同语义）
    if early_stopping.is_some() && best_round + 1 < trees[0].len() {
        for class_trees in &mut trees {
            class_trees.truncate(best_round + 1);
        }
    }

    let best_iteration = trees[0].len();
    Ok(MulticlassBooster {
        n_classes,
        trees,
        table,
        init_scores,
        learning_rate: params.learning_rate,
        best_iteration,
        eval_history,
    })
}

/// 早停验证集的解析状态（内部；列 clone 为拥有，规避自引用借用）。
struct EvalState {
    labels: Vec<usize>,
    cols: Vec<Float64Array>,
    policy: crate::data::MissingPolicy,
}

/// 每行 softmax（数值稳定，减 max）。
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// 校验标签 ∈ [0, n_classes) 且为整数。
fn to_labels(y: &[f64], n_classes: usize) -> Result<Vec<usize>, BoostingError> {
    y.iter()
        .map(|&v| {
            if v.fract() != 0.0 || !(0.0..n_classes as f64).contains(&v) {
                Err(BoostingError::Data(DataError::InvalidLabel {
                    value: v,
                    n_classes,
                }))
            } else {
                Ok(v as usize)
            }
        })
        .collect()
}

fn predict_row(tree: &Tree, cols: &[&Float64Array], row: usize, policy: MissingPolicy) -> f64 {
    tree.predict_one(|f| {
        let col = cols[f];
        let v = col.value(row);
        (v, is_missing_value(v, col.is_null(row), policy))
    })
}

fn feature_columns(ds: &Dataset) -> Result<Vec<&Float64Array>, BoostingError> {
    let mut cols = Vec::with_capacity(ds.num_features());
    for f in 0..ds.num_features() {
        cols.push(ds.feature_values(f)?);
    }
    Ok(cols)
}
