use std::{error::Error, sync::Arc};

use ::redis::aio::{ConnectionLike, ConnectionManager};

use super::store::{is_field_name, MetadataField, RedisDistance, Store, VectorAlgorithm};
use crate::embedding::Embedder;

/// Builds a Redis [`Store`].
///
/// ```rust,ignore
/// let store = StoreBuilder::new()
///     .url("redis://127.0.0.1/")
///     .index_name("docs")
///     .embedder(OpenAiEmbedder::default())
///     .build()
///     .await?;
/// ```
pub struct StoreBuilder {
    url: String,
    index_name: Option<String>,
    key_prefix: Option<String>,
    embedder: Option<Arc<dyn Embedder>>,
    content_key: String,
    vector_key: String,
    namespace_key: String,
    namespace: Option<String>,
    metadata_schema: Vec<MetadataField>,
    distance: RedisDistance,
    algorithm: VectorAlgorithm,
}

impl Default for StoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StoreBuilder {
    pub fn new() -> Self {
        Self {
            url: "redis://127.0.0.1:6379/".to_string(),
            index_name: None,
            key_prefix: None,
            embedder: None,
            content_key: "content".to_string(),
            vector_key: "content_vector".to_string(),
            namespace_key: "namespace".to_string(),
            namespace: None,
            metadata_schema: Vec::new(),
            distance: RedisDistance::default(),
            algorithm: VectorAlgorithm::default(),
        }
    }

    /// Redis URL used by [`build`](Self::build). Default: `redis://127.0.0.1:6379/`.
    pub fn url(mut self, url: &str) -> Self {
        self.url = url.to_string();
        self
    }

    /// Name of the RediSearch index (required).
    pub fn index_name(mut self, index_name: &str) -> Self {
        self.index_name = Some(index_name.to_string());
        self
    }

    /// Prefix of the document keys. Default: `doc:{index_name}:`.
    pub fn key_prefix(mut self, key_prefix: &str) -> Self {
        self.key_prefix = Some(key_prefix.to_string());
        self
    }

    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    /// Hash field holding the text. Default: `content`.
    pub fn content_key(mut self, key: &str) -> Self {
        self.content_key = key.to_string();
        self
    }

    /// Hash field holding the embedding. Default: `content_vector`.
    pub fn vector_key(mut self, key: &str) -> Self {
        self.vector_key = key.to_string();
        self
    }

    /// TAG field holding the namespace. Default: `namespace`.
    pub fn namespace_key(mut self, key: &str) -> Self {
        self.namespace_key = key.to_string();
        self
    }

    /// Default namespace for writes and searches; `VecStoreOptions::name_space` overrides it.
    pub fn namespace(mut self, namespace: &str) -> Self {
        self.namespace = Some(namespace.to_string());
        self
    }

    /// Metadata fields to index, so filters can use them. Other metadata is still stored
    /// and returned, just not searchable. Only applies when the store creates the index.
    pub fn metadata_schema(mut self, fields: impl IntoIterator<Item = MetadataField>) -> Self {
        self.metadata_schema = fields.into_iter().collect();
        self
    }

    /// Distance metric of a new index. Default: cosine.
    pub fn distance(mut self, distance: RedisDistance) -> Self {
        self.distance = distance;
        self
    }

    /// Vector algorithm of a new index. Default: HNSW.
    pub fn algorithm(mut self, algorithm: VectorAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Connects to [`url`](Self::url) with a `ConnectionManager`, which reconnects on its own.
    pub async fn build(self) -> Result<Store, Box<dyn Error>> {
        let client = ::redis::Client::open(self.url.as_str())?;
        let connection = ConnectionManager::new(client).await?;
        self.build_with_connection(connection)
    }

    /// Uses an existing async connection (`MultiplexedConnection`, a cluster connection...).
    pub fn build_with_connection<C>(self, connection: C) -> Result<Store<C>, Box<dyn Error>>
    where
        C: ConnectionLike + Clone + Send + Sync,
    {
        let index_name = self.index_name.ok_or("index_name is required")?;
        let embedder = self.embedder.ok_or("embedder is required")?;
        let names = [&self.content_key, &self.vector_key, &self.namespace_key]
            .into_iter()
            .chain(self.metadata_schema.iter().map(|f| &f.name));
        for name in names {
            if !is_field_name(name) {
                return Err(format!(
                    "invalid Redis field name {name:?}: use letters, digits and _"
                )
                .into());
            }
        }
        for (i, field) in self.metadata_schema.iter().enumerate() {
            let reserved = [&self.content_key, &self.vector_key, &self.namespace_key];
            if reserved.contains(&&field.name)
                || self.metadata_schema[..i]
                    .iter()
                    .any(|f| f.name == field.name)
            {
                return Err(format!("duplicate Redis field {:?} in the schema", field.name).into());
            }
        }
        Ok(Store {
            connection,
            embedder,
            key_prefix: self
                .key_prefix
                .unwrap_or_else(|| format!("doc:{index_name}:")),
            index_name,
            content_key: self.content_key,
            vector_key: self.vector_key,
            namespace_key: self.namespace_key,
            namespace: self.namespace,
            metadata_schema: self.metadata_schema,
            distance: self.distance,
            algorithm: self.algorithm,
            index_ready: Default::default(),
        })
    }
}
