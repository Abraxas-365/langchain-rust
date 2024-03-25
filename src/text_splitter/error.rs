use thiserror::Error;

#[derive(Error, Debug)]
pub enum TextSplitterError {
    #[error("Empty input text")]
    EmptyInputText,

    #[error("Mismatch metadata and text")]
    MetadataTextMismatch,

    #[error("Tokenizer not found")]
    TokenizerNotFound,

    #[error("Tokenizer creation failed due to invalid tokenizer")]
    InvalidTokenizer,

    #[error("Tokenizer creation failed due to invalid model")]
    InvalidModel,

    #[error("Error: {0}")]
    OtherError(String),
}
