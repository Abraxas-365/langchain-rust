use thiserror::Error;

#[derive(Error, Debug)]
pub enum MiniMaxError {
    #[error("MiniMax API error: Invalid Format - {0}")]
    InvalidFormatError(String),

    #[error("MiniMax API error: Authentication Failed - {0}")]
    AuthenticationError(String),

    #[error("MiniMax API error: Insufficient Balance - {0}")]
    InsufficientBalanceError(String),

    #[error("MiniMax API error: Invalid Parameters - {0}")]
    InvalidParametersError(String),

    #[error("MiniMax API error: Rate Limit Reached - {0}")]
    RateLimitError(String),

    #[error("MiniMax API error: Server Error - {0}")]
    ServerError(String),

    #[error("MiniMax API error: Server Overloaded - {0}")]
    ServerOverloadedError(String),
}
