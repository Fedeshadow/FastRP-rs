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
            Self::Oom(err) => {
                let dbg_str = format!("{:?}", err);
                // Attempt to parse "size: XXXXX," from the debug string since TryReserveError
                // does not expose the layout/size directly in stable Rust yet.
                let req_mem_msg = if let Some(start) = dbg_str.find("size: ") {
                    let rest = &dbg_str[start + 6..];
                    if let Some(end) = rest.find(',') {
                        if let Ok(bytes) = rest[..end].parse::<usize>() {
                            let mb = bytes as f64 / 1_048_576.0;
                            format!("Requested Memory: {:.2} MB", mb)
                        } else {
                            "Requested Memory: Unknown".to_string()
                        }
                    } else {
                        "Requested Memory: Unknown".to_string()
                    }
                } else {
                    "Requested Memory: Unknown".to_string()
                };

                write!(
                    f,
                    "Out Of Memory (OOM): The system could not allocate enough memory.\n\
                     - {}\n\
                     Try reducing the number of nodes or the embedding dimension.\n\n\
                     Raw Error:\n{}",
                    req_mem_msg, dbg_str
                )
            }
            Self::ShapeMismatch(msg) => write!(f, "Shape mismatch: {}", msg),
        }
    }
}

impl From<TryReserveError> for FastRPError {
    fn from(err: TryReserveError) -> Self {
        Self::Oom(err)
    }
}
