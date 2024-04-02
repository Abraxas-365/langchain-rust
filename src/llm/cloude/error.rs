use thiserror::Error;

#[derive(Error, Debug)]
pub enum AnthropicError {
    #[error("Anthropic API error: Invalid request - {0}")]
    InvalidRequestError(String),

    #[error("Anthropic API error: Authentication failed - {0}")]
    AuthenticationError(String),

    #[error("Anthropic API error: Permission denied - {0}")]
    PermissionError(String),

    #[error("Anthropic API error: Not found - {0}")]
    NotFoundError(String),

    #[error("Anthropic API error: Rate limit exceeded - {0}")]
    RateLimitError(String),

    #[error("Anthropic API error: Internal error - {0}")]
    ApiError(String),

    #[error("Anthropic API error: Overloaded - {0}")]
    OverloadedError(String),
}
