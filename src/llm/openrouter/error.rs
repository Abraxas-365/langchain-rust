
//! OpenRouter LLM errors.

use thiserror::Error;

/// Errors specific to OpenRouter LLM API.
#[derive(Debug, Error, Clone)]
pub enum OpenRouterError {
    /// API key is missing from environment or configuration.
    #[error("OpenRouter API key not set")]
    ApiKeyMissing,

    /// HTTP 400: The request was invalid or malformed.
    #[error("OpenRouter API Bad Request (400): {0}")]
    BadRequest(String),

    /// HTTP 401: Authentication failed due to missing or invalid API key.
    #[error("OpenRouter API Unauthorized (401): {0}")]
    Unauthorized(String),

    /// HTTP 402: Payment required, insufficient credits or quota.
    #[error("OpenRouter API Payment Required (402): {0}")]
    PaymentRequired(String),

    /// HTTP 403: Access to the requested resource is forbidden.
    #[error("OpenRouter API Forbidden (403): {0}")]
    Forbidden(String),

    /// HTTP 404: The requested resource or model was not found.
    #[error("OpenRouter API Not Found (404): {0}")]
    NotFound(String),

    /// HTTP 429: Rate limit exceeded.
    #[error("OpenRouter API Rate Limit (429): {0}")]
    RateLimit(String),

    /// HTTP 500: Internal server error on OpenRouter API.
    #[error("OpenRouter API Internal Server Error (500): {0}")]
    InternalServerError(String),

    /// HTTP 502: Bad gateway from OpenRouter API.
    #[error("OpenRouter API Bad Gateway (502): {0}")]
    BadGateway(String),

    /// HTTP 503: Service unavailable (maintenance or overload).
    #[error("OpenRouter API Service Unavailable (503): {0}")]
    ServiceUnavailable(String),

    /// HTTP 504: Gateway timeout from upstream model provider.
    #[error("OpenRouter API Gateway Timeout (504): {0}")]
    GatewayTimeout(String),

    /// Request failed (network or protocol error).
    #[error("OpenRouter request failed: {0}")]
    RequestFailed(String),

    /// OpenRouter API returned a non-specific error.
    #[error("OpenRouter API returned error: {0}")]
    ApiError(String),

    /// Streaming is not yet implemented.
    #[error("OpenRouter streaming not implemented")]
    NotImplemented,

    /// Unknown or uncategorized error.
    #[error("OpenRouter: unknown error")]
    Unknown(String),
}
