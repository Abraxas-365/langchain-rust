//! [llmman](https://github.com/llmmanorg/llmman) is a local model runner that serves the
//! Ollama API (alongside OpenAI- and Anthropic-compatible ones) on port 17434.
//!
//! The wire protocol is identical to Ollama's, so this module reuses the Ollama
//! integration and only changes the default endpoint.
//!
//! ## Example
//!
//! ```rs
//! let llmman = Ollama::llmman().with_model("gemma4");
//! let response = llmman.invoke("Say hello!").await.unwrap();
//! ```

#[cfg(feature = "ollama")]
use crate::embedding::ollama::OllamaEmbedder;
#[cfg(feature = "ollama")]
use crate::llm::ollama::client::{Ollama, OllamaClient};
use crate::llm::ollama::openai::OllamaConfig;
#[cfg(feature = "ollama")]
use std::sync::Arc;

/// Default llmman endpoint. Override with the `LLMMAN_HOST` env var (`[host][:port]`).
pub const LLMMAN_API_BASE: &str = "http://localhost:17434";

/// Base URL of the llmman server, honouring `LLMMAN_HOST` (`[host][:port]`).
pub fn llmman_api_base() -> String {
    match std::env::var("LLMMAN_HOST") {
        Ok(host) if !host.is_empty() => {
            if host.contains("://") {
                return host.trim_end_matches('/').to_string();
            }
            let (host, port) = match host.rsplit_once(':') {
                Some((h, p)) if p.chars().all(|c| c.is_ascii_digit()) => (h, p),
                _ => (host.as_str(), "17434"),
            };
            let host = if host.is_empty() { "localhost" } else { host };
            format!("http://{host}:{port}")
        }
        _ => LLMMAN_API_BASE.to_string(),
    }
}

#[cfg(feature = "ollama")]
fn llmman_client() -> Arc<OllamaClient> {
    Arc::new(OllamaClient::try_new(llmman_api_base()).expect("invalid LLMMAN_HOST"))
}

#[cfg(feature = "ollama")]
impl Ollama {
    /// Creates an [`Ollama`] LLM backed by a local llmman server (see [`llmman_api_base`]).
    /// Panics if `LLMMAN_HOST` is not a valid URL.
    pub fn llmman() -> Self {
        Ollama::new(llmman_client(), "gemma4", None)
    }
}

#[cfg(feature = "ollama")]
impl OllamaEmbedder {
    /// Creates an [`OllamaEmbedder`] backed by a local llmman server (see [`llmman_api_base`]).
    /// Panics if `LLMMAN_HOST` is not a valid URL.
    pub fn llmman() -> Self {
        OllamaEmbedder::new(llmman_client(), "nomic-embed-text", None)
    }
}

impl OllamaConfig {
    /// OpenAI-compatible config for a local llmman server (see [`llmman_api_base`]).
    pub fn llmman() -> Self {
        Self::default().with_api_base(format!("{}/v1", llmman_api_base()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::language_models::llm::LLM;

    #[test]
    fn test_llmman_api_base() {
        std::env::remove_var("LLMMAN_HOST");
        assert_eq!(llmman_api_base(), LLMMAN_API_BASE);
        for (host, expected) in [
            (":18000", "http://localhost:18000"),
            ("myhost", "http://myhost:17434"),
            ("0.0.0.0:17434", "http://0.0.0.0:17434"),
            ("https://example.com/", "https://example.com"),
        ] {
            std::env::set_var("LLMMAN_HOST", host);
            assert_eq!(llmman_api_base(), expected);
        }
        std::env::remove_var("LLMMAN_HOST");
    }

    #[tokio::test]
    #[ignore]
    async fn test_llmman_openai() {
        let llmman = crate::llm::openai::OpenAI::new(OllamaConfig::llmman()).with_model("gemma4");
        let response = llmman.invoke("hola").await.unwrap();
        println!("{}", response);
    }

    #[cfg(feature = "ollama")]
    #[tokio::test]
    #[ignore]
    async fn test_llmman_generate() {
        let llmman = Ollama::llmman().with_model("gemma4");
        let response = llmman.invoke("Hey Macarena, ay").await.unwrap();
        println!("{}", response);
    }

    #[cfg(feature = "ollama")]
    #[tokio::test]
    #[ignore]
    async fn test_llmman_embed() {
        use crate::embedding::Embedder;
        let response = OllamaEmbedder::llmman()
            .embed_query("Why is the sky blue?")
            .await
            .unwrap();
        assert_eq!(response.len(), 768);
    }
}
