use async_trait::async_trait;

use super::{OutputParser, OutputParserError};

pub struct SimpleParser {
    trim: bool,
}
impl SimpleParser {
    pub fn new() -> Self {
        Self { trim: false }
    }
    pub fn with_trim(mut self, trim: bool) -> Self {
        self.trim = trim;
        self
    }
}
impl Default for SimpleParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OutputParser for SimpleParser {
    async fn parse(&self, output: &str) -> Result<String, OutputParserError> {
        if self.trim {
            Ok(output.trim().to_string())
        } else {
            Ok(output.to_string())
        }
    }
}
