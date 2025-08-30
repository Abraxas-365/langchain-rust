use async_openai::config::Config;
use reqwest;
use secrecy::ExposeSecret;
use serde::{Deserialize, Serialize};

use crate::llm::nanogpt::NanoGPT;

#[derive(Debug, Serialize, Deserialize)]
pub struct ModelResponse {
    pub object: String,
    pub data: Vec<Model>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pricing: Option<Pricing>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Pricing {
    pub prompt: f64,
    pub completion: f64,
    pub currency: String,
    pub unit: String,
}

impl<C: Config + Clone> NanoGPT<C> {
    pub async fn get_models(&self, detailed: bool) -> Result<ModelResponse, reqwest::Error> {
        let client = reqwest::Client::new();
        let url = self.config.url("/models");
        let mut request = client.get(&url);

        if detailed {
            request = request.query(&[("detailed", "true")]);
        }

        request = request.header(
            "Authorization",
            format!("Bearer {}", self.config.api_key().expose_secret()),
        );

        let response = request.send().await?;
        response.json().await
    }
}
