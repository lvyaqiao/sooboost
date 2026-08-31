//! 热替换（contracts §1.2 修订：std `Arc` 无 `swap`，单写者发布用 `RwLock<Arc<..>>`）。
//!
//! 读者 `load()` 取 Arc 快照（读锁仅覆盖 Arc 克隆，预测本体无锁）；
//! 写者 `publish()` 原子替换；在途读者仍持旧 Arc，读旧模型安全。
//! 读重场景再评估 arc-swap（RCU 等价物），不预埋。

use std::sync::{Arc, RwLock};

use crate::boosting::Booster;
use crate::loss::Loss;

/// 可热替换模型槽（单写者发布、多读者快照）。
#[derive(Debug)]
pub struct HotSwappable<L: Loss> {
    current: RwLock<Arc<Booster<L>>>,
}

impl<L: Loss> HotSwappable<L> {
    pub fn new(model: Booster<L>) -> Self {
        Self {
            current: RwLock::new(Arc::new(model)),
        }
    }

    /// 原子读快照（Arc 克隆；之后对新发布不可见，但旧模型仍有效）。
    pub fn load(&self) -> Arc<Booster<L>> {
        let guard = self.current.read().unwrap_or_else(|p| p.into_inner());
        Arc::clone(&guard)
    }

    /// 原子发布新模型（在途快照不受影响）。
    pub fn publish(&self, model: Booster<L>) {
        let new_arc = Arc::new(model);
        let mut guard = self.current.write().unwrap_or_else(|p| p.into_inner());
        *guard = new_arc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Dataset, MissingPolicy};
    use crate::loss::SquaredError;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc as StdArc;

    fn tiny_booster(y: Vec<f64>, n_estimators: usize) -> Booster<SquaredError> {
        let schema = Schema::new(vec![
            Field::new("f0", DataType::Float64, true),
            Field::new("target", DataType::Float64, false),
        ]);
        let batch = RecordBatch::try_new(
            StdArc::new(schema),
            vec![
                StdArc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                StdArc::new(Float64Array::from(y)),
            ],
        )
        .expect("batch");
        let ds = Dataset::from_record_batch(batch, &["f0"], "target", MissingPolicy::default())
            .expect("dataset");
        crate::boosting::fit(
            &ds,
            &crate::boosting::BoostingParams {
                n_estimators,
                ..crate::boosting::BoostingParams::default()
            },
            SquaredError,
            &crate::boosting::TrainingContext::new(1),
        )
        .expect("fit")
    }

    #[test]
    fn publish_swaps_and_old_snapshot_still_works() {
        let h = HotSwappable::new(tiny_booster(vec![1.0, 2.0, 3.0, 4.0], 5));
        let old = h.load();
        let pred_before = old.predict_row(&[1.0], &[false]);

        let new_model = tiny_booster(vec![1.0, 2.0, 3.0, 4.0], 50);
        let pred_new_expected = new_model.predict_row(&[1.0], &[false]);
        h.publish(new_model);

        let cur = h.load();
        assert_eq!(cur.num_trees(), 50);
        assert_eq!(
            cur.predict_row(&[1.0], &[false]),
            pred_new_expected,
            "发布后新模型生效"
        );
        // 在途旧快照仍可预测（旧模型有效）
        assert_eq!(old.predict_row(&[1.0], &[false]), pred_before);
    }
}
