use serde_json::Value;
use std::io::{self, Write};

#[derive(Debug, Clone)]
pub struct StreamData {
    pub value: Value,
    pub content: String,
}
impl StreamData {
    pub fn new<S: Into<String>>(value: Value, content: S) -> Self {
        Self {
            value,
            content: content.into(),
        }
    }

    pub fn to_stdout(&self) -> io::Result<()> {
        let stdout = io::stdout();
        let mut handle = stdout.lock();
        write!(handle, "{}", self.content)?;
        handle.flush()
    }
}
