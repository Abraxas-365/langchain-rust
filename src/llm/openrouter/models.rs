
//! OpenRouter supported model definitions.

use serde::{Deserialize, Serialize};

/// Enum of popular OpenRouter model string identifiers.
#[derive(Debug, Clone)]
pub enum OpenRouterModel {
    /// OpenAI GPT-4.1
    Gpt41,
    /// OpenAI GPT-4.1-mini
    Gpt41Mini,
    /// OpenAI GPT-4.1-nano
    Gpt41Nano,
    /// OpenAI GPT-3.5 Turbo
    Gpt35Turbo,
    /// OpenAI GPT-4o
    Gpt4o,

    /// Google Gemini 2.5 Pro Preview
    Gemini25ProPreview,

    /// Anthropic Claude 3 Haiku
    Claude3Haiku,
    /// Anthropic Claude 3 Sonnet
    Claude3Sonnet,
    /// Anthropic Claude 3 Opus
    Claude3Opus,


    /// Custom model by string in the format manufacturer/model_name
    Custom(String),
}

impl OpenRouterModel {
    /// Get the string identifier for this model.
    pub fn as_str(&self) -> &str {
        match self {
            OpenRouterModel::Gpt41 => "openai/gpt-4.1",
            OpenRouterModel::Gpt41Mini => "openai/gpt-4.1-mini",
            OpenRouterModel::Gpt41Nano => "openai/gpt-4.1-nano",
            OpenRouterModel::Gpt35Turbo => "openai/gpt-3.5-turbo",
            OpenRouterModel::Gpt4o => "openai/gpt-4o",

            OpenRouterModel::Gemini25ProPreview => "google/gemini-2.5-pro-preview-03-25",

            OpenRouterModel::Claude3Haiku => "anthropic/claude-3-haiku-20240307",
            OpenRouterModel::Claude3Sonnet => "anthropic/claude-3-sonnet-20240229",
            OpenRouterModel::Claude3Opus => "anthropic/claude-3-opus-20240229",
            OpenRouterModel::Custom(s) => s.as_str(),
        }
    }

    /// Lists all available models via the OpenRouter API.
    ///
    /// # Arguments
    /// * `api_key` - Your OpenRouter API key.
    ///
    /// # Returns
    /// A vector of ModelInfo structs representing all models, or an OpenRouterError.
    pub async fn list_available_models(
        api_key: &str,
    ) -> Result<Vec<ModelInfo>, crate::llm::openrouter::error::OpenRouterError> {
        use reqwest::Client;
        use crate::llm::openrouter::error::OpenRouterError;

        let client = Client::new();
        let resp = client
            .get("https://openrouter.ai/api/v1/models")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| OpenRouterError::RequestFailed(e.to_string()))?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| OpenRouterError::RequestFailed(e.to_string()))?;

        if !status.is_success() {
            return Err(OpenRouterError::ApiError(format!(
                "HTTP {}: {}",
                status, text
            )));
        }

        let resp_json: ModelListResponse = serde_json::from_str(&text)
            .map_err(|e| OpenRouterError::RequestFailed(format!("Invalid JSON: {}", e)))?;

        Ok(resp_json.data)
    }

    /// Gets details for a specific model via the OpenRouter API.
    ///
    /// # Arguments
    /// * `api_key` - Your OpenRouter API key.
    /// * `model_id` - The model identifier string.
    ///
    /// # Returns
    /// ModelInfo for the specified model, or an OpenRouterError.
    pub async fn get_model_details(
        api_key: &str,
        model_id: &str,
    ) -> Result<ModelInfo, crate::llm::openrouter::error::OpenRouterError> {
        use reqwest::Client;
        use crate::llm::openrouter::error::OpenRouterError;

        let client = Client::new();
        let url = format!("https://openrouter.ai/api/v1/models/{}", model_id);
        let resp = client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .send()
            .await
            .map_err(|e| OpenRouterError::RequestFailed(e.to_string()))?;

        let status = resp.status();
        let text = resp.text().await.map_err(|e| OpenRouterError::RequestFailed(e.to_string()))?;

        if !status.is_success() {
            return Err(OpenRouterError::ApiError(format!(
                "HTTP {}: {}",
                status, text
            )));
        }

        let model_info: ModelInfo = serde_json::from_str(&text)
            .map_err(|e| OpenRouterError::RequestFailed(format!("Invalid JSON: {}", e)))?;

        Ok(model_info)
    }
}

/// Model pricing info for OpenRouter models.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelPricing {
    /// Price per 1M prompt tokens (USD)
    pub prompt: Option<f64>,
    /// Price per 1M completion tokens (USD)
    pub completion: Option<f64>,
}

/// OpenRouter model metadata as returned by the API.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelInfo {
    /// Model identifier string.
    pub id: String,
    /// Human-readable model name.
    pub name: String,
    /// Optional model description.
    pub description: Option<String>,
    /// Optional pricing info.
    pub pricing: Option<ModelPricing>,
}

/// For deserializing list response from /api/v1/models
#[derive(Debug, Clone, Deserialize)]
struct ModelListResponse {
    pub data: Vec<ModelInfo>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Integration test for model listing (requires real API key).
    #[tokio::test]
    #[ignore]
    async fn test_list_available_models_real_api() {
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            eprintln!("OPENROUTER_API_KEY not set, skipping test.");
            return;
        }
        let result = OpenRouterModel::list_available_models(&api_key).await;
        assert!(result.is_ok(), "Failed to list models: {:?}", result);
        let models = result.unwrap();
        assert!(!models.is_empty(), "Model list should not be empty");
        eprintln!("Got {} models", models.len());
    }

    /// Integration test for model details (requires real API key and valid model_id).
    #[tokio::test]
    #[ignore]
    async fn test_get_model_details_real_api() {
        let api_key = std::env::var("OPENROUTER_API_KEY").unwrap_or_default();
        if api_key.is_empty() {
            eprintln!("OPENROUTER_API_KEY not set, skipping test.");
            return;
        }
        // You may need to update this model_id to a valid one in your account.
        let model_id = "openai/gpt-4o";
        let result = OpenRouterModel::get_model_details(&api_key, model_id).await;
        assert!(result.is_ok(), "Failed to get model details: {:?}", result);
        let info = result.unwrap();
        assert_eq!(info.id, model_id);
        eprintln!("Model info: {:?}", info);
    }
}
