use thiserror::Error;

#[derive(Error, Debug)]
pub enum TextSplitterError {
    #[error("Empty input text")]
    EmptyInputText,

    #[error("Mismatch metadata and text")]
    MetadataTextMismatch,

    #[error("Tokenizer not found")]
    TokenizerNotFound,

    #[error("Tokenizer creation failed: {0}")]
    TokenizerCreationFailed(#[from] anyhow::Error),
}
