use std::collections::TryReserveError;

#[derive(Debug)]
pub enum FastRPError {
    /// Caught an allocation failure gracefully
    Oom(TryReserveError),
    /// For future customization
    ShapeMismatch(String),
}

impl std::error::Error for FastRPError {}

impl std::fmt::Display for FastRPError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Oom(err) => write!(f, "Allocation failed due to out of memory constraints: {}", err),
            Self::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
        }
    }
}

impl From<TryReserveError> for FastRPError {
    fn from(err: TryReserveError) -> Self {
        Self::Oom(err)
    }
}
