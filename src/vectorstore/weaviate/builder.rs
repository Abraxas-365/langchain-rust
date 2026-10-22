use std::{error::Error, sync::Arc};

use reqwest::Client;

use super::store::{is_graphql_name, Store};
use crate::embedding::Embedder;

/// Builds a Weaviate [`Store`].
///
/// ```rust,ignore
/// let store = StoreBuilder::new()
///     .url("http://localhost:8080")
///     .class_name("Document")
///     .embedder(OpenAiEmbedder::default())
///     .build()?;
/// ```
pub struct StoreBuilder {
    client: Option<Client>,
    url: String,
    api_key: Option<String>,
    class_name: Option<String>,
    embedder: Option<Arc<dyn Embedder>>,
    text_key: String,
    namespace_key: String,
    namespace: Option<String>,
    metadata_fields: Option<Vec<String>>,
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
            url: "http://localhost:8080".to_string(),
            api_key: None,
            class_name: None,
            embedder: None,
            text_key: "text".to_string(),
            namespace_key: "nameSpace".to_string(),
            namespace: None,
            metadata_fields: None,
        }
    }

    /// Weaviate base URL. Default: `http://localhost:8080`.
    pub fn url(mut self, url: &str) -> Self {
        self.url = url.trim_end_matches('/').to_string();
        self
    }

    /// API key, sent as a bearer token (Weaviate Cloud or API-key auth). Default: none.
    pub fn api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }

    /// A preconfigured HTTP client (timeouts, proxies...).
    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Collection (class) name. REQUIRED. Weaviate capitalizes the first letter, and so
    /// does this builder.
    pub fn class_name(mut self, name: &str) -> Self {
        self.class_name = Some(name.to_string());
        self
    }

    /// Embeddings provider. REQUIRED.
    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    /// Property holding the document text. Default: `text` (same as langchaingo).
    pub fn text_key(mut self, key: &str) -> Self {
        self.text_key = key.to_string();
        self
    }

    /// Property holding the namespace. Default: `nameSpace` (same as langchaingo).
    pub fn namespace_key(mut self, key: &str) -> Self {
        self.namespace_key = key.to_string();
        self
    }

    /// Default namespace when the options don't set `name_space`. Default: none, so
    /// documents aren't scoped.
    pub fn namespace(mut self, namespace: &str) -> Self {
        self.namespace = Some(namespace.to_string());
        self
    }

    /// Properties returned as metadata. Default: every scalar property in the collection
    /// schema, which is fetched on each search.
    pub fn metadata_fields<S: Into<String>>(mut self, fields: impl IntoIterator<Item = S>) -> Self {
        self.metadata_fields = Some(fields.into_iter().map(Into::into).collect());
        self
    }

    pub fn build(self) -> Result<Store, Box<dyn Error>> {
        let class_name = self.class_name.ok_or("'class_name' is required")?;
        let embedder = self.embedder.ok_or("'embedder' is required")?;
        let mut chars = class_name.chars();
        let class_name = match chars.next() {
            Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
            None => String::new(),
        };
        for (what, name) in [
            ("class name", &class_name),
            ("text_key", &self.text_key),
            ("namespace_key", &self.namespace_key),
        ] {
            if !is_graphql_name(name) {
                return Err(format!("invalid Weaviate {what} {name:?}").into());
            }
        }
        Ok(Store {
            client: self.client.unwrap_or_default(),
            url: self.url,
            api_key: self.api_key,
            class_name,
            embedder,
            text_key: self.text_key,
            namespace_key: self.namespace_key,
            namespace: self.namespace,
            metadata_fields: self.metadata_fields,
            schema_ready: Default::default(),
        })
    }
}
