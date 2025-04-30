use thiserror::Error;

#[derive(Error, Debug)]
pub enum QwenError {
    #[error("Qwen API error: Invalid parameter - {0}")]
    InvalidParameterError(String),

    #[error("Qwen API error: Invalid API Key - {0}")]
    InvalidApiKeyError(String),

    #[error("Qwen API error: Network error - {0}")]
    NetworkError(String),

    #[error("Qwen API error: Model Unavailable - {0}")]
    ModelUnavailableError(String),

    #[error("Qwen API error: Rate limit exceeded - {0}")]
    ModelServingError(String),

    #[error("Qwen API error: Internal error - {0}")]
    InternalError(String),

    #[error("Qwen API error: System error - {0}")]
    SystemError(String),

    #[error("Qwen API error: Billing issue - {0}")]
    BillingError(String),

    #[error("Qwen API error: Mismatched model - {0}")]
    MismatchedModelError(String),

    #[error("Qwen API error: Duplicate custom ID - {0}")]
    DuplicateCustomIdError(String),

    #[error("Qwen API error: Model not found - {0}")]
    ModelNotFoundError(String),

    #[error("Qwen API error: Connection error - {0}")]
    APIConnectionError(String),

    #[error("Qwen API error: Prepaid bill overdue - {0}")]
    PrepaidBillOverdueError(String),

    #[error("Qwen API error: Postpaid bill overdue - {0}")]
    PostpaidBillOverdueError(String),

    #[error("Qwen API error: Commodity not purchased - {0}")]
    CommodityNotPurchasedError(String),

    #[error("Qwen API error: Internal algorithm error - {0}")]
    InternalAlgorithmError(String),

    #[error("Qwen API error: Timeout - {0}")]
    TimeoutError(String),

    #[error("Qwen API error: Rewrite failed - {0}")]
    RewriteFailedError(String),

    #[error("Qwen API error: Retrieval failed - {0}")]
    RetrievalFailedError(String),

    #[error("Qwen API error: Application process failed - {0}")]
    AppProcessFailedError(String),

    #[error("Qwen API error: Model service failed - {0}")]
    ModelServiceFailedError(String),

    #[error("Qwen API error: Plugin invocation failed - {0}")]
    InvokePluginFailedError(String),
}
