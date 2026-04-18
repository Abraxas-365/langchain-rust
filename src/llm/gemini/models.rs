use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The request structure for the Gemini API `generateContent` and `streamGenerateContent` endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiGenerateRequest {
    /// The conversation history and current user input.
    pub contents: Vec<GeminiContent>,
    /// Optional system instructions applied to the entire conversation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_instruction: Option<GeminiContent>,
    /// Optional configuration for text generation (temperature, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation_config: Option<GenerationConfig>,
    /// Optional safety settings to bypass or configure thresholds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safety_settings: Option<Vec<Value>>,
}

/// A structured container for a message role and its text or media parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiContent {
    /// The role of the author. Normally "user" or "model".
    pub role: String,
    /// The pieces of data forming this message (e.g., text strings).
    pub parts: Vec<GeminiPart>,
}

/// A single piece of content within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeminiPart {
    /// The raw text string for this part.
    pub text: String,
}

/// Configuration settings controlling the generation randomness and bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationConfig {
    /// Sampling temperature. Higher values produce more random text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// The maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Top-P sampling (nucleus sampling).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-K sampling.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

/// The response payload received from the Gemini API text endpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GeminiResponse {
    /// Discovered candidates. Normally a list of length 1 containing the model's reply.
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    /// Statistical information regarding token usage for billing and debugging.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
}

/// A single generated output from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Candidate {
    /// The actual content text strings returned by the model.
    pub content: GeminiContent,
    /// Reason why the generation ended (e.g., "STOP").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Safety ratings provided by the internal safety filters.
    #[serde(default)]
    pub safety_ratings: Vec<SafetyRating>,
}

/// The safety classification provided by Google for the given candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SafetyRating {
    /// Category of the filter (e.g., "HARM_CATEGORY_HATE_SPEECH").
    pub category: String,
    /// Assessed severity level (e.g., "NEGLIGIBLE").
    pub probability: String,
    /// Indicates whether the content was actually blocked.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked: Option<bool>,
}

/// Token usage metadata detailing request limits and token consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UsageMetadata {
    /// Tokens consumed by the prompt text.
    pub prompt_token_count: u32,
    /// Tokens consumed by the output generation.
    pub candidates_token_count: u32,
    /// Total combined tokens for billing.
    #[serde(default)]
    pub total_token_count: u32,
}
