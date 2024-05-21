use async_openai::config::Config;
use reqwest::header::HeaderMap;
use secrecy::Secret;
use serde::Deserialize;

const OLLAMA_API_BASE: &str = "http://localhost:11434/v1";

/// Ollama has [OpenAI compatiblity](https://ollama.com/blog/openai-compatibility), meaning that you can use it as an OpenAI API.
///
/// This struct implements the `Config` trait of OpenAI, and has the necessary setup for OpenAI configurations for you to use Ollama.
///
/// ## Example
///
/// ```rs
/// let ollama = OpenAI::new(OllamaConfig::default()).with_model("llama3");
/// let response = ollama.invoke("Say hello!").await.unwrap();
/// ```
#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct OllamaConfig {
    api_key: Secret<String>,
}

impl OllamaConfig {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Config for OllamaConfig {
    fn api_key(&self) -> &Secret<String> {
        &self.api_key
    }

    fn api_base(&self) -> &str {
        OLLAMA_API_BASE
    }

    fn headers(&self) -> HeaderMap {
        HeaderMap::default()
    }

    fn query(&self) -> Vec<(&str, &str)> {
        vec![]
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.api_base(), path)
    }
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            api_key: Secret::new("ollama".to_string()),
        }
    }
}

#[cfg(tests)]
mod tests {
    use crate::{
        language_models::llm::LLM,
        llm::{ollama::openai::OllamaConfig, openai::OpenAI},
    };

    #[tokio::test]
    async fn test_ollama_openai() {
        use super::*;

        let ollama = OpenAI::new(OllamaConfig::default()).with_model("llama2");
        let response = ollama.invoke("hola").await.unwrap();
        println!("{}", response);
    }
}
