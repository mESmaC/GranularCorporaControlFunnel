use serde::Serialize;

#[derive(Debug, thiserror::Error)]
pub enum GccfError {
    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[error("arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),
    #[error("{0}")]
    Msg(String),
    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

impl GccfError {
    pub fn msg(s: impl Into<String>) -> Self {
        GccfError::Msg(s.into())
    }
}

// Tauri needs command errors to be serializable so the frontend receives a string.
impl Serialize for GccfError {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

pub type Result<T> = std::result::Result<T, GccfError>;
