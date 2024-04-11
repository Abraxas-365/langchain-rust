use async_trait::async_trait;
use regex::Regex;

use super::{OutputParser, OutputParserError};

pub struct MarkdownParser {
    expresion: String,
    trim: bool,
}
impl MarkdownParser {
    pub fn new() -> Self {
        Self {
            expresion: r"```(?:\w+)?\s*([\s\S]+?)\s*```".to_string(),
            trim: false,
        }
    }

    pub fn with_custom_expresion(mut self, expresion: &str) -> Self {
        self.expresion = expresion.to_string();
        self
    }

    pub fn with_trim(mut self, trim: bool) -> Self {
        self.trim = trim;
        self
    }
}
impl Default for MarkdownParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl OutputParser for MarkdownParser {
    async fn parse(&self, output: &str) -> Result<String, OutputParserError> {
        let re = Regex::new(r"```(?:\w+)?\s*([\s\S]+?)\s*```")?;
        if let Some(cap) = re.captures(output) {
            let find = cap[1].to_string();
            if self.trim {
                Ok(find.trim().to_string())
            } else {
                Ok(find)
            }
        } else {
            Err(OutputParserError::ParsingError(
                "No code block found".into(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_markdown_parser_finds_code_block() {
        let parser = MarkdownParser::new();
        let markdown_content = r#"
```rust
fn main() {
    println!("Hello, world!");
}
```
"#;
        let result = parser.parse(markdown_content).await;
        println!("{:?}", result);

        let correct = r#"fn main() {
    println!("Hello, world!");
}"#;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), correct);
    }
}
