use async_openai::error::OpenAIError;
#[cfg(feature = "ollama")]
use ollama_rs::error::OllamaError;
use reqwest::{Error as ReqwestError, StatusCode};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbedderError {
    #[error("Network request failed: {0}")]
    RequestError(#[from] ReqwestError),

    #[error("OpenAI error: {0}")]
    OpenAIError(#[from] OpenAIError),

    #[error("URL parsing error: {0}")]
    UrlParseError(#[from] url::ParseError),

    #[error("HTTP error: {status_code} {error_message}")]
    HttpError {
        status_code: StatusCode,
        error_message: String,
    },

    #[error("FastEmbed error: {0}")]
    FastEmbedError(String),

    #[cfg(feature = "ollama")]
    #[error("Ollama error: {0}")]
    OllamaError(#[from] OllamaError),
}
