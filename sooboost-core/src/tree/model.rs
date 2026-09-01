//! 推理优先树表示（SoA 紧凑数组，架构 D7）。
//!
//! 布局：`split_features` / `thresholds` / `missing_go_left` / `left` / `right` /
//! `leaf_values` 全部按节点索引对齐；叶子节点的 `left[i] == -1`。
//! 分裂语义：`x <= threshold → 左子树`；缺失走 `missing_go_left` 指定的方向。

/// 一棵二叉决策树（标量叶子）。
#[derive(Debug, Clone)]
pub struct Tree {
    split_features: Vec<usize>,
    thresholds: Vec<f64>,
    missing_go_left: Vec<bool>,
    left: Vec<i32>,
    right: Vec<i32>,
    leaf_values: Vec<f64>,
    depths: Vec<usize>,
    /// 分裂增益（每节点；叶子为 0.0）。v3 格式起随模型持久化，供特征重要度（gain）。
    split_gains: Vec<f64>,
    /// 节点覆盖样本数（每节点；建树时该节点分区行数）。v3 格式起持久化，供特征重要度（cover）。
    node_counts: Vec<f64>,
}

impl Tree {
    pub fn num_nodes(&self) -> usize {
        self.left.len()
    }

    /// 树最大深度（根=0）。
    pub fn max_depth(&self) -> usize {
        self.depths.iter().copied().max().unwrap_or(0)
    }

    /// 单行推断：`get(feature)` 返回 (特征值, 是否缺失)。
    pub fn predict_one(&self, mut get: impl FnMut(usize) -> (f64, bool)) -> f64 {
        let mut i = 0usize;
        loop {
            if self.left[i] < 0 {
                return self.leaf_values[i];
            }
            let f = self.split_features[i];
            let (x, missing) = get(f);
            let go_left = if missing {
                self.missing_go_left[i]
            } else {
                x <= self.thresholds[i]
            };
            i = if go_left { self.left[i] } else { self.right[i] } as usize;
        }
    }

    /// 单行推断（数组输入）：`values[f]` / `is_missing[f]`。
    pub fn predict(&self, values: &[f64], is_missing: &[bool]) -> f64 {
        self.predict_one(|f| (values[f], is_missing[f]))
    }

    // -- 序列化所需访问器（model/ 模块用，只读） --------------------------

    pub(crate) fn split_features(&self) -> &[usize] {
        &self.split_features
    }

    pub(crate) fn thresholds(&self) -> &[f64] {
        &self.thresholds
    }

    pub(crate) fn missing_go_left(&self) -> &[bool] {
        &self.missing_go_left
    }

    pub(crate) fn left(&self) -> &[i32] {
        &self.left
    }

    pub(crate) fn right(&self) -> &[i32] {
        &self.right
    }

    pub(crate) fn leaf_values(&self) -> &[f64] {
        &self.leaf_values
    }

    pub(crate) fn depths(&self) -> &[usize] {
        &self.depths
    }

    /// 分裂增益（每节点；叶子为 0.0）。
    pub(crate) fn split_gains(&self) -> &[f64] {
        &self.split_gains
    }

    /// 节点覆盖样本数（每节点）。
    pub(crate) fn node_counts(&self) -> &[f64] {
        &self.node_counts
    }

    /// 由已校验的 SoA 数组构造（model/ 反序列化用；字段合法性由调用方保证）。
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_soa(
        split_features: Vec<usize>,
        thresholds: Vec<f64>,
        missing_go_left: Vec<bool>,
        left: Vec<i32>,
        right: Vec<i32>,
        leaf_values: Vec<f64>,
        depths: Vec<usize>,
        split_gains: Vec<f64>,
        node_counts: Vec<f64>,
    ) -> Self {
        Self {
            split_features,
            thresholds,
            missing_go_left,
            left,
            right,
            leaf_values,
            depths,
            split_gains,
            node_counts,
        }
    }

    /// 由构建缓冲节点组装 SoA（内部使用）。
    pub(crate) fn from_nodes(nodes: Vec<super::builder::NodeBuf>) -> Self {
        let len = nodes.len();
        let mut split_features = vec![0usize; len];
        let mut thresholds = vec![0.0; len];
        let mut missing_go_left = vec![false; len];
        let mut left = vec![-1i32; len];
        let mut right = vec![-1i32; len];
        let mut leaf_values = vec![0.0; len];
        let mut depths = vec![0usize; len];
        let mut split_gains = vec![0.0; len];
        let mut node_counts = vec![0.0; len];

        for (i, node) in nodes.into_iter().enumerate() {
            node_counts[i] = (node.range.1 - node.range.0) as f64;
            if let Some(f) = node.split_feature {
                split_features[i] = f;
                thresholds[i] = node.threshold;
                missing_go_left[i] = node.missing_go_left;
                if let (Some(l), Some(r)) = (node.left, node.right) {
                    left[i] = l as i32;
                    right[i] = r as i32;
                }
                depths[i] = node.depth;
                split_gains[i] = node.gain;
            }
            leaf_values[i] = node.leaf_value.unwrap_or(0.0);
        }

        Self {
            split_features,
            thresholds,
            missing_go_left,
            left,
            right,
            leaf_values,
            depths,
            split_gains,
            node_counts,
        }
    }
}
