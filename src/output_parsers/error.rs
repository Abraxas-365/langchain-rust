use regex::Error as RegexError;
use thiserror::Error;

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum OutputParserError {
    #[error("Regex error: {0}")]
    RegexError(#[from] RegexError),

    #[error("Parsing error: {0}")]
    ParsingError(String),
}
