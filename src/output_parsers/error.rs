use regex::Error as RegexError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum OutputParserError {
    #[error("Regex error: {0}")]
    RegexError(#[from] RegexError),

    #[error("Parsing error: {0}")]
    ParsingError(String),
}
