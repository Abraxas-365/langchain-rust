use std::{error::Error, sync::Arc, time::Duration};

use reqwest::Client;

use super::store::Store;
use crate::embedding::Embedder;

const METRICS: &[&str] = &["COSINE", "IP", "L2"];
const CONSISTENCY_LEVELS: &[&str] = &["Strong", "Session", "Bounded", "Eventually"];

/// Builds a Milvus [`Store`].
///
/// ```rust,ignore
/// let store = StoreBuilder::new()
///     .url("http://localhost:19530")
///     .collection("documents")
///     .embedder(OpenAiEmbedder::default())
///     .build()?;
/// ```
pub struct StoreBuilder {
    client: Option<Client>,
    url: String,
    token: Option<String>,
    db_name: Option<String>,
    collection: Option<String>,
    embedder: Option<Arc<dyn Embedder>>,
    primary_field: String,
    text_field: String,
    vector_field: String,
    metadata_field: String,
    namespace_field: String,
    namespace: Option<String>,
    metric: String,
    consistency_level: String,
    max_text_length: usize,
    create_collection: bool,
    load_timeout: Duration,
}

impl Default for StoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Milvus field and collection names: a letter or `_`, then letters, digits or `_`.
pub(super) fn is_milvus_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
        && name.len() <= 255
}

impl StoreBuilder {
    pub fn new() -> Self {
        StoreBuilder {
            client: None,
            url: "http://localhost:19530".to_string(),
            token: None,
            db_name: None,
            collection: None,
            embedder: None,
            primary_field: "pk".to_string(),
            text_field: "text".to_string(),
            vector_field: "vector".to_string(),
            metadata_field: "metadata".to_string(),
            namespace_field: "namespace".to_string(),
            namespace: None,
            metric: "COSINE".to_string(),
            consistency_level: "Strong".to_string(),
            max_text_length: 65_535,
            create_collection: true,
            load_timeout: Duration::from_secs(60),
        }
    }

    /// Milvus (or Zilliz Cloud) endpoint. Default: `http://localhost:19530`.
    pub fn url(mut self, url: &str) -> Self {
        self.url = url.trim_end_matches('/').to_string();
        self
    }

    /// Auth token, sent as a bearer token: `user:password` with authentication on, or a
    /// Zilliz Cloud API key. Default: none.
    pub fn token(mut self, token: &str) -> Self {
        self.token = Some(token.to_string());
        self
    }

    /// Database. Default: Milvus' `default`.
    pub fn db_name(mut self, db_name: &str) -> Self {
        self.db_name = Some(db_name.to_string());
        self
    }

    /// A preconfigured HTTP client (timeouts, proxies...).
    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Collection name. REQUIRED.
    pub fn collection(mut self, name: &str) -> Self {
        self.collection = Some(name.to_string());
        self
    }

    /// Embeddings provider. REQUIRED.
    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    /// Primary key of collections this store creates. Default: `pk`. Existing
    /// collections use their own primary key.
    pub fn primary_field(mut self, name: &str) -> Self {
        self.primary_field = name.to_string();
        self
    }

    /// `VarChar` field holding the document text. Default: `text`.
    pub fn text_field(mut self, name: &str) -> Self {
        self.text_field = name.to_string();
        self
    }

    /// `FloatVector` field. Default: `vector`.
    pub fn vector_field(mut self, name: &str) -> Self {
        self.vector_field = name.to_string();
        self
    }

    /// `JSON` field holding the metadata. Default: `metadata`.
    pub fn metadata_field(mut self, name: &str) -> Self {
        self.metadata_field = name.to_string();
        self
    }

    /// `VarChar` field holding the namespace (the partition key in created collections).
    /// Default: `namespace`.
    pub fn namespace_field(mut self, name: &str) -> Self {
        self.namespace_field = name.to_string();
        self
    }

    /// Default namespace when the options don't set `name_space`. Default: none.
    pub fn namespace(mut self, namespace: &str) -> Self {
        self.namespace = Some(namespace.to_string());
        self
    }

    /// Metric of collections this store creates: `COSINE` (default), `IP` or `L2`.
    /// Existing collections use their index's metric.
    pub fn metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_ascii_uppercase();
        self
    }

    /// Consistency level of created collections: `Strong` (default, writes are searchable
    /// at once), `Session`, `Bounded` or `Eventually`.
    pub fn consistency_level(mut self, level: &str) -> Self {
        self.consistency_level = level.to_string();
        self
    }

    /// `max_length` of the text field in created collections (bytes, up to 65,535).
    pub fn max_text_length(mut self, max: usize) -> Self {
        self.max_text_length = max;
        self
    }

    /// Create the collection on the first write if it's missing. Default: `true`.
    pub fn create_collection(mut self, create: bool) -> Self {
        self.create_collection = create;
        self
    }

    /// How long to wait for a collection to load. Default: 60s.
    pub fn load_timeout(mut self, timeout: Duration) -> Self {
        self.load_timeout = timeout;
        self
    }

    pub fn build(self) -> Result<Store, Box<dyn Error>> {
        let collection = self.collection.ok_or("'collection' is required")?;
        let embedder = self.embedder.ok_or("'embedder' is required")?;
        let names = [
            ("collection", &collection),
            ("primary_field", &self.primary_field),
            ("text_field", &self.text_field),
            ("vector_field", &self.vector_field),
            ("metadata_field", &self.metadata_field),
            ("namespace_field", &self.namespace_field),
        ];
        for (what, name) in names {
            if !is_milvus_name(name) {
                return Err(format!("invalid Milvus {what} {name:?}").into());
            }
        }
        let fields = &names[1..];
        for (i, (what, name)) in fields.iter().enumerate() {
            if let Some((other, _)) = fields[..i].iter().find(|(_, n)| n == name) {
                return Err(format!("{what} and {other} are both {name:?}").into());
            }
        }
        if !METRICS.contains(&self.metric.as_str()) {
            return Err(format!(
                "unsupported Milvus metric {:?}, use one of {METRICS:?}",
                self.metric
            )
            .into());
        }
        if !CONSISTENCY_LEVELS.contains(&self.consistency_level.as_str()) {
            return Err(format!(
                "unknown Milvus consistency level {:?}, use one of {CONSISTENCY_LEVELS:?}",
                self.consistency_level
            )
            .into());
        }
        if !(1..=65_535).contains(&self.max_text_length) {
            return Err("max_text_length must be between 1 and 65535".into());
        }
        Ok(Store {
            client: self.client.unwrap_or_default(),
            url: self.url,
            token: self.token,
            db_name: self.db_name,
            collection,
            embedder,
            primary_field: self.primary_field,
            text_field: self.text_field,
            vector_field: self.vector_field,
            metadata_field: self.metadata_field,
            namespace_field: self.namespace_field,
            namespace: self.namespace,
            metric: self.metric,
            consistency_level: self.consistency_level,
            max_text_length: self.max_text_length,
            create_collection: self.create_collection,
            load_timeout: self.load_timeout,
            schema: Default::default(),
        })
    }
}
