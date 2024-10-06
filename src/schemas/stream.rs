use serde_json::Value;
use std::io::{self, Write};

use crate::language_models::TokenUsage;

#[derive(Debug, Clone)]
pub struct StreamData {
    pub value: Value,
    pub tokens: Option<TokenUsage>,
    pub content: String,
}

impl StreamData {
    pub fn new<S: Into<String>>(value: Value, tokens: Option<TokenUsage>, content: S) -> Self {
        Self {
            value,
            tokens,
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
