use thiserror::Error;

#[derive(Error, Debug)]
pub enum DeepseekError {
    #[error("Deepseek API error: Invalid Format - {0}")]
    InvalidFormatError(String),

    #[error("Deepseek API error: Authentication Failed - {0}")]
    AuthenticationError(String),

    #[error("Deepseek API error: Insufficient Balance - {0}")]
    InsufficientBalanceError(String),

    #[error("Deepseek API error: Invalid Parameters - {0}")]
    InvalidParametersError(String),

    #[error("Deepseek API error: Rate Limit Reached - {0}")]
    RateLimitError(String),

    #[error("Deepseek API error: Server Error - {0}")]
    ServerError(String),

    #[error("Deepseek API error: Server Overloaded - {0}")]
    ServerOverloadedError(String),
} 