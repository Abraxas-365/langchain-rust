use std::{error::Error, sync::Arc};

use reqwest::Client;
use serde_json::{json, Value};

use super::store::{send, ChromaDistance, Store};
use crate::embedding::Embedder;

/// Builds a Chroma [`Store`], creating the collection if it doesn't exist.
///
/// ```rust,ignore
/// let store = StoreBuilder::new()
///     .url("http://localhost:8000")
///     .collection_name("docs")
///     .embedder(OpenAiEmbedder::default())
///     .build()
///     .await?;
/// ```
pub struct StoreBuilder {
    client: Option<Client>,
    url: String,
    tenant: String,
    database: String,
    api_key: Option<String>,
    collection_name: Option<String>,
    embedder: Option<Arc<dyn Embedder>>,
    distance: ChromaDistance,
    namespace_key: String,
    collection_metadata: Option<Value>,
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
            url: "http://localhost:8000".to_string(),
            tenant: "default_tenant".to_string(),
            database: "default_database".to_string(),
            api_key: None,
            collection_name: None,
            embedder: None,
            distance: ChromaDistance::default(),
            namespace_key: "namespace".to_string(),
            collection_metadata: None,
        }
    }

    /// Chroma server URL. Default: `http://localhost:8000`.
    pub fn url(mut self, url: &str) -> Self {
        self.url = url.trim_end_matches('/').to_string();
        self
    }

    /// Tenant. Default: `default_tenant`.
    pub fn tenant(mut self, tenant: &str) -> Self {
        self.tenant = tenant.to_string();
        self
    }

    /// Database. Default: `default_database`.
    pub fn database(mut self, database: &str) -> Self {
        self.database = database.to_string();
        self
    }

    /// API key, sent as the `x-chroma-token` header (Chroma Cloud or token auth).
    pub fn api_key(mut self, api_key: &str) -> Self {
        self.api_key = Some(api_key.to_string());
        self
    }

    /// A preconfigured HTTP client (timeouts, proxies, extra headers...).
    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Collection name. REQUIRED. Created if it doesn't exist.
    pub fn collection_name(mut self, collection_name: &str) -> Self {
        self.collection_name = Some(collection_name.to_string());
        self
    }

    /// Embeddings provider. REQUIRED.
    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    /// Distance function used when the collection is created. Default: cosine.
    /// An existing collection keeps its own; set this to match it so scores are right.
    pub fn distance(mut self, distance: ChromaDistance) -> Self {
        self.distance = distance;
        self
    }

    /// Metadata key that `VecStoreOptions::name_space` writes and filters on.
    /// Default: `namespace`.
    pub fn namespace_key(mut self, key: &str) -> Self {
        self.namespace_key = key.to_string();
        self
    }

    /// Metadata stored on the collection when it's created.
    pub fn collection_metadata(mut self, metadata: Value) -> Self {
        self.collection_metadata = Some(metadata);
        self
    }

    /// Gets or creates the collection and returns the store.
    pub async fn build(self) -> Result<Store, Box<dyn Error>> {
        let collection_name = self
            .collection_name
            .ok_or("'collection_name' is required")?;
        let embedder = self.embedder.ok_or("'embedder' is required")?;
        let client = self.client.unwrap_or_default();

        let collections_url = format!(
            "{}/api/v2/tenants/{}/databases/{}/collections",
            self.url, self.tenant, self.database
        );
        let mut body = json!({
            "name": collection_name,
            "get_or_create": true,
            "configuration": { "hnsw": { "space": self.distance.as_str() } },
        });
        if let Some(metadata) = self.collection_metadata {
            body["metadata"] = metadata;
        }
        let collection = send(client.post(&collections_url).json(&body), &self.api_key).await?;
        let id = collection["id"]
            .as_str()
            .ok_or("Chroma didn't return a collection id")?;

        Ok(Store {
            client,
            collection_url: format!("{collections_url}/{id}"),
            api_key: self.api_key,
            embedder,
            distance: self.distance,
            namespace_key: self.namespace_key,
        })
    }
}
