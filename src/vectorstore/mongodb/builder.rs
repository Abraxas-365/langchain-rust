use std::{error::Error, sync::Arc, time::Duration};

use mongodb::{bson::Document as BsonDocument, Client, Collection};

use super::store::{MongoSimilarity, Store, SCORE_FIELD};
use crate::embedding::Embedder;

/// Builds a MongoDB Atlas [`Store`].
///
/// ```rust,ignore
/// let store = StoreBuilder::new()
///     .uri("mongodb+srv://user:pass@cluster0.example.mongodb.net/")
///     .database("app")
///     .collection("docs")
///     .embedder(OpenAiEmbedder::default())
///     .build()
///     .await?;
/// ```
pub struct StoreBuilder {
    uri: String,
    database: Option<String>,
    collection: Option<String>,
    embedder: Option<Arc<dyn Embedder>>,
    index_name: String,
    text_key: String,
    embedding_key: String,
    metadata_key: String,
    namespace_key: String,
    namespace: Option<String>,
    filter_paths: Vec<String>,
    similarity: MongoSimilarity,
    num_candidates: usize,
    create_index: bool,
    index_timeout: Duration,
}

impl Default for StoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StoreBuilder {
    pub fn new() -> Self {
        Self {
            uri: "mongodb://localhost:27017/".to_string(),
            database: None,
            collection: None,
            embedder: None,
            index_name: "vector_index".to_string(),
            text_key: "text".to_string(),
            embedding_key: "embedding".to_string(),
            metadata_key: "metadata".to_string(),
            namespace_key: "namespace".to_string(),
            namespace: None,
            filter_paths: Vec::new(),
            similarity: MongoSimilarity::default(),
            num_candidates: 10,
            create_index: true,
            index_timeout: Duration::from_secs(120),
        }
    }

    /// Connection string used by [`build`](Self::build).
    /// Default: `mongodb://localhost:27017/`.
    pub fn uri(mut self, uri: &str) -> Self {
        self.uri = uri.to_string();
        self
    }

    /// Database name (required by [`build`](Self::build)).
    pub fn database(mut self, database: &str) -> Self {
        self.database = Some(database.to_string());
        self
    }

    /// Collection name (required by [`build`](Self::build)).
    pub fn collection(mut self, collection: &str) -> Self {
        self.collection = Some(collection.to_string());
        self
    }

    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    /// Name of the Atlas `vectorSearch` index. Default: `vector_index`.
    pub fn index_name(mut self, name: &str) -> Self {
        self.index_name = name.to_string();
        self
    }

    /// Field holding the text. Default: `text`.
    pub fn text_key(mut self, key: &str) -> Self {
        self.text_key = key.to_string();
        self
    }

    /// Field holding the embedding. Default: `embedding`.
    pub fn embedding_key(mut self, key: &str) -> Self {
        self.embedding_key = key.to_string();
        self
    }

    /// Sub-document holding the metadata. Default: `metadata`.
    pub fn metadata_key(mut self, key: &str) -> Self {
        self.metadata_key = key.to_string();
        self
    }

    /// Field holding the namespace. Default: `namespace`.
    pub fn namespace_key(mut self, key: &str) -> Self {
        self.namespace_key = key.to_string();
        self
    }

    /// Default namespace for writes and searches; `VecStoreOptions::name_space` overrides it.
    pub fn namespace(mut self, namespace: &str) -> Self {
        self.namespace = Some(namespace.to_string());
        self
    }

    /// Document paths to index as filters (e.g. `metadata.source`), so `filters` can use
    /// them. The namespace field is always included. Only applies when the store creates
    /// the index.
    pub fn filter_paths<S: Into<String>>(mut self, paths: impl IntoIterator<Item = S>) -> Self {
        self.filter_paths = paths.into_iter().map(Into::into).collect();
        self
    }

    /// Similarity of a new index. Default: cosine.
    pub fn similarity(mut self, similarity: MongoSimilarity) -> Self {
        self.similarity = similarity;
        self
    }

    /// `numCandidates` is `limit × multiplier`, never below `limit` and capped at 10,000.
    /// Higher is more accurate and slower. Default: 10, as Atlas recommends.
    pub fn num_candidates_multiplier(mut self, multiplier: usize) -> Self {
        self.num_candidates = multiplier;
        self
    }

    /// Whether the first write creates the index if it's missing. Default: true. Turn it
    /// off when the index is managed elsewhere (Terraform, the Atlas UI...).
    pub fn create_index(mut self, create: bool) -> Self {
        self.create_index = create;
        self
    }

    /// How long the first write waits for a new index to become queryable.
    /// Default: 2 minutes.
    pub fn index_timeout(mut self, timeout: Duration) -> Self {
        self.index_timeout = timeout;
        self
    }

    /// Connects to [`uri`](Self::uri).
    pub async fn build(self) -> Result<Store, Box<dyn Error>> {
        let database = self.database.clone().ok_or("database is required")?;
        let collection = self.collection.clone().ok_or("collection is required")?;
        let client = Client::with_uri_str(&self.uri).await?;
        let collection = client.database(&database).collection(&collection);
        self.build_with_collection(collection)
    }

    /// Uses an existing collection handle (shared client, custom options...).
    pub fn build_with_collection(
        self,
        collection: Collection<BsonDocument>,
    ) -> Result<Store, Box<dyn Error>> {
        let embedder = self.embedder.ok_or("embedder is required")?;
        let keys = [
            &self.text_key,
            &self.embedding_key,
            &self.metadata_key,
            &self.namespace_key,
        ];
        for key in keys {
            if key.is_empty() || key.starts_with('$') || key.contains('.') || key == "_id" {
                return Err(format!(
                    "invalid MongoDB field {key:?}: use a top-level name, not _id"
                )
                .into());
            }
        }
        for (i, key) in keys.iter().enumerate() {
            if keys[..i].contains(key) || *key == SCORE_FIELD {
                return Err(format!("MongoDB field {key:?} is used twice").into());
            }
        }
        Ok(Store {
            collection,
            embedder,
            index_name: self.index_name,
            text_key: self.text_key,
            embedding_key: self.embedding_key,
            metadata_key: self.metadata_key,
            namespace_key: self.namespace_key,
            namespace: self.namespace,
            filter_paths: self.filter_paths,
            similarity: self.similarity,
            num_candidates: self.num_candidates,
            create_index: self.create_index,
            index_timeout: self.index_timeout,
            index_ready: Default::default(),
        })
    }
}
