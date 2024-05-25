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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{language_models::llm::LLM, llm::openai::OpenAI, schemas::Message};
    use tokio::io::AsyncWriteExt;
    use tokio_stream::StreamExt;

    #[tokio::test]
    #[ignore]
    async fn test_ollama_openai() {
        let ollama = OpenAI::new(OllamaConfig::default()).with_model("llama2");
        let response = ollama.invoke("hola").await.unwrap();
        println!("{}", response);
    }

    #[tokio::test]
    #[ignore]
    async fn test_ollama_openai_stream() {
        let ollama = OpenAI::new(OllamaConfig::default()).with_model("phi3");

        let message = Message::new_human_message("Why does water boil at 100 degrees?");
        let mut stream = ollama.stream(&vec![message]).await.unwrap();
        let mut stdout = tokio::io::stdout();
        while let Some(res) = stream.next().await {
            let data = res.unwrap();
            stdout.write(data.content.as_bytes()).await.unwrap();
        }
        stdout.flush().await.unwrap();
    }
}
