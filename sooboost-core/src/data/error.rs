//! 数据层错误类型：全部显式传播（易踩坑 10：错误处理显式化）。

use arrow::datatypes::DataType;

/// 数据层错误。
#[derive(Debug, thiserror::Error)]
pub enum DataError {
    /// 引用的列名不存在。
    #[error("列 `{0}` 不存在")]
    ColumnNotFound(String),

    /// 列类型非 Float64（M0 仅支持数值特征；M1 起支持 Dictionary 类别特征）。
    #[error("列 `{name}` 类型为 {ty:?}，仅支持 Float64 数值列或 Dictionary 类别列")]
    UnsupportedType { name: String, ty: DataType },

    /// 类别列字典键类型不支持（仅 UInt16/UInt32）。
    #[error("类别列 `{name}` 字典键类型 {ty:?} 不支持（仅 UInt16/UInt32）")]
    UnsupportedDictionary { name: String, ty: DataType },

    /// 类别特征被当作数值特征读取（用错 API）。
    #[error("特征 `{0}` 是类别特征，请用 categorical_key 读取")]
    CategoricalFeatureNotNumeric(String),

    /// 类别基数超过上限（超限报错而非静默截断，contracts §1.4）。
    #[error("特征 `{name}` 类别基数 {got} 超过上限 {limit}")]
    TooManyCategories {
        name: String,
        got: usize,
        limit: usize,
    },

    /// 同一列同时出现在特征列与 target 列。
    #[error("特征列与 target 列重叠: `{0}`")]
    FeatureTargetOverlap(String),

    /// 数据集为空（0 行），无法训练。
    #[error("数据集为空（0 行），无法训练")]
    EmptyDataset,

    /// 多分类类别数非法（< 2）。
    #[error("多分类类别数 {0} 非法（需 ≥ 2）")]
    InvalidMulticlassClasses(usize),

    /// 多分类标签非法（非整数或越界）。
    #[error("多分类标签 {value} 非法（需整数且 ∈ [0, {n_classes})）")]
    InvalidLabel { value: f64, n_classes: usize },

    /// 拼接 schema 不一致（Dataset::concatenate_rows，M6-2）。
    #[error("数据集拼接失败: {reason}")]
    ConcatSchemaMismatch {
        /// 不一致原因。
        reason: &'static str,
    },

    /// 行切片越界（Dataset::slice_rows，M6-2）。
    #[error("行切片越界: offset={offset} length={length} 超过总行数 {rows}")]
    RowSliceOutOfBounds {
        /// 起始行。
        offset: usize,
        /// 切片长度。
        length: usize,
        /// 总行数。
        rows: usize,
    },

    /// 底层 arrow 错误。
    #[error("arrow 错误: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// IO 错误。
    #[error("IO 错误: {0}")]
    Io(#[from] std::io::Error),
}
