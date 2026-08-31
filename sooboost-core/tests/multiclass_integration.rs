//! 多分类集成测试：softmax GBDT 训练/预测回路 + 标签校验（D8 M1-2）。

use arrow::array::{Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

use sooboost_core::boosting::{BoostingParams, TrainingContext, fit_multiclass};
use sooboost_core::data::{DataError, Dataset, MissingPolicy};

/// 三簇可分数据：f0 中心分别 ≈0 / 5 / 10，标签 0/1/2。
fn make_dataset() -> Dataset {
    let mut f0 = Vec::new();
    let mut target = Vec::new();
    for (center, label) in [(0.0, 0), (5.0, 1), (10.0, 2)] {
        for k in 0..200 {
            f0.push(center + ((k % 10) as f64) * 0.01);
            target.push(label);
        }
    }
    let schema = Schema::new(vec![
        Field::new("f0", DataType::Float64, true),
        Field::new("target", DataType::Int64, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float64Array::from(f0)),
            Arc::new(Int64Array::from(target)),
        ],
    )
    .expect("batch");
    Dataset::from_record_batch(batch, &["f0"], "target", MissingPolicy::default()).expect("dataset")
}

#[test]
fn fit_and_predict_separable_classes() {
    let ds = make_dataset();
    let model = fit_multiclass(
        &ds,
        &BoostingParams {
            n_estimators: 20,
            ..BoostingParams::default()
        },
        3,
        &TrainingContext::new(1),
    )
    .expect("fit multiclass");

    assert_eq!(model.n_classes(), 3);
    assert_eq!(model.num_trees_per_class(), 20);
    assert_eq!(model.init_scores().len(), 3);

    let proba = model.predict_proba(&ds).expect("proba");
    assert_eq!(proba.len(), ds.num_rows());
    for row in &proba {
        assert_eq!(row.len(), 3);
        let sum: f64 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "softmax 每行和为 1，实际 {sum}");
        // 可分离数据下 softmax 可饱和到 0/1，故用闭区间
        assert!(row.iter().all(|p| (0.0..=1.0).contains(p)));
    }

    let preds = model.predict(&ds).expect("predict");
    let labels: Vec<usize> = ds
        .target_values()
        .values()
        .iter()
        .map(|&v| v as usize)
        .collect();
    let acc =
        preds.iter().zip(&labels).filter(|(p, l)| *p == *l).count() as f64 / preds.len() as f64;
    assert!(acc > 0.95, "可分数据多分类训练准确率应 > 0.95，实际 {acc}");
}

#[test]
fn fewer_than_two_classes_is_error() {
    let ds = make_dataset();
    let err =
        fit_multiclass(&ds, &BoostingParams::default(), 1, &TrainingContext::new(1)).unwrap_err();
    assert!(matches!(
        err,
        sooboost_core::boosting::BoostingError::Data(DataError::InvalidMulticlassClasses(1))
    ));
}

#[test]
fn non_integer_or_out_of_range_label_is_error() {
    let schema = Schema::new(vec![
        Field::new("f0", DataType::Float64, true),
        Field::new("target", DataType::Float64, false),
    ]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 3.0])),
            Arc::new(Float64Array::from(vec![0.0, 1.0, 2.0, 5.0])), // 5 越界（n_classes=3）
        ],
    )
    .expect("batch");
    let ds = Dataset::from_record_batch(batch, &["f0"], "target", MissingPolicy::default())
        .expect("dataset");
    let err =
        fit_multiclass(&ds, &BoostingParams::default(), 3, &TrainingContext::new(1)).unwrap_err();
    assert!(matches!(
        err,
        sooboost_core::boosting::BoostingError::Data(DataError::InvalidLabel { value, n_classes })
            if value == 5.0 && n_classes == 3
    ));
}
