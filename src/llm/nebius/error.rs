use thiserror::Error;

#[derive(Error, Debug)]
pub enum NebiusError {
    #[error("Nebius API error: Invalid request - {0}")]
    InvalidRequestError(String),

    #[error("Nebius API error: Authentication failed - {0}")]
    AuthenticationError(String),

    #[error("Nebius API error: Permission denied - {0}")]
    PermissionError(String),

    #[error("Nebius API error: Not found - {0}")]
    NotFoundError(String),

    #[error("Nebius API error: Rate limit exceeded - {0}")]
    RateLimitError(String),

    #[error("Nebius API error: Internal error - {0}")]
    ApiError(String),

    #[error("Nebius API error: Service unavailable - {0}")]
    ServiceUnavailableError(String),
}
