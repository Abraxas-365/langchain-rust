use std::{error::Error, sync::Arc};

use reqwest::Client;

use super::store::Store;
use crate::embedding::Embedder;

pub const DEFAULT_PINECONE_API_VERSION: &str = "2025-10";

/// Builds a Pinecone [`Store`] for an existing index.
///
/// ```rust,ignore
/// let store = StoreBuilder::new()
///     .host("https://my-index-abc123.svc.aped-1234.pinecone.io") // from the console or describe_index
///     .api_key(&std::env::var("PINECONE_API_KEY")?)
///     .embedder(OpenAiEmbedder::default())
///     .build()?;
/// ```
pub struct StoreBuilder {
    client: Option<Client>,
    host: Option<String>,
    api_key: Option<String>,
    api_version: String,
    embedder: Option<Arc<dyn Embedder>>,
    namespace: String,
    text_key: String,
}

impl Default for StoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StoreBuilder {
    pub fn new() -> Self {
        StoreBuilder {
            client: None,
            host: None,
            api_key: None,
            api_version: DEFAULT_PINECONE_API_VERSION.to_string(),
            embedder: None,
            namespace: String::new(),
            text_key: "text".to_string(),
        }
    }

    /// Index host, e.g. `https://my-index-abc123.svc.aped-1234.pinecone.io`. REQUIRED.
    /// `https://` is added when missing.
    pub fn host(mut self, host: &str) -> Self {
        let host = host.trim_end_matches('/');
        self.host = Some(if host.contains("://") {
            host.to_string()
        } else {
            format!("https://{host}")
        });
        self
    }

    /// API key. Default: the `PINECONE_API_KEY` environment variable.
    pub fn api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }

    /// Value of the `X-Pinecone-Api-Version` header. Default: [`DEFAULT_PINECONE_API_VERSION`].
    pub fn api_version(mut self, version: &str) -> Self {
        self.api_version = version.to_string();
        self
    }

    /// A preconfigured HTTP client (timeouts, proxies...).
    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Embeddings provider. REQUIRED.
    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    /// Default namespace for writes and searches. Default: `""` (the default namespace).
    pub fn namespace(mut self, namespace: &str) -> Self {
        self.namespace = namespace.to_string();
        self
    }

    /// Metadata key holding the document text. Default: `text`.
    pub fn text_key(mut self, key: &str) -> Self {
        self.text_key = key.to_string();
        self
    }

    pub fn build(self) -> Result<Store, Box<dyn Error>> {
        let host = self.host.ok_or("'host' is required")?;
        let embedder = self.embedder.ok_or("'embedder' is required")?;
        let api_key = match self.api_key {
            Some(key) => key,
            None => std::env::var("PINECONE_API_KEY")
                .map_err(|_| "'api_key' is required (or set PINECONE_API_KEY)")?,
        };
        Ok(Store {
            client: self.client.unwrap_or_default(),
            host,
            api_key,
            api_version: self.api_version,
            embedder,
            namespace: self.namespace,
            text_key: self.text_key,
        })
    }
}
