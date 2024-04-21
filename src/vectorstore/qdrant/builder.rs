use crate::embedding::Embedder;
use crate::vectorstore::qdrant::Store;
use qdrant_client::client::QdrantClient;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{CreateCollection, Distance, Filter, VectorParams, VectorsConfig};
use std::error::Error;
use std::sync::Arc;

pub struct StoreBuilder {
    client: Option<QdrantClient>,
    embedder: Option<Arc<dyn Embedder>>,
    collection_name: Option<String>,
    content_field: String,
    metadata_field: String,
    recreate_collection: bool,
    search_filter: Option<Filter>,
}

impl Default for StoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StoreBuilder {
    /// Create a new StoreBuilder object with default values.
    pub fn new() -> Self {
        StoreBuilder {
            client: None,
            embedder: None,
            collection_name: None,
            search_filter: None,
            content_field: "page_content".to_string(),
            metadata_field: "metadata".to_string(),
            recreate_collection: false,
        }
    }

    /// An instance of `qdrant_client::QdrantClient` for the Store. REQUIRED.
    pub fn client(mut self, client: QdrantClient) -> Self {
        self.client = Some(client);
        self
    }

    /// Embeddings provider for the Store. REQUIRED.
    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    /// Name of the collection in Qdrant. REQUIRED.
    /// It is recommended to create a collection in advance, with the required configurations.
    /// https://qdrant.tech/documentation/concepts/collections/#create-a-collection
    ///
    /// If the collection doesn't exist, it will be created with the embedding provider's dimension
    /// and Cosine similarity metric.
    pub fn collection_name(mut self, collection_name: &str) -> Self {
        self.collection_name = Some(collection_name.to_string());
        self
    }

    /// Name of the field in the Qdrant point's payload that will store the metadata of the documents.
    /// Default: "metadata"
    pub fn metadata_field(mut self, metadata_field: &str) -> Self {
        self.metadata_field = metadata_field.to_string();
        self
    }

    /// Name of the field in the Qdrant point's payload that will store the content of the documents.
    /// Default: "page_content"
    pub fn content_field(mut self, content_field: &str) -> Self {
        self.content_field = content_field.to_string();
        self
    }

    /// If set to true, the collection will be deleted and recreated using
    /// the embedding provider's dimension and Cosine similarity metric.
    pub fn recreate_collection(mut self, recreate_collection: bool) -> Self {
        self.recreate_collection = recreate_collection;
        self
    }

    /// Filter to be applied to the search results.
    /// https://qdrant.tech/documentation/concepts/filtering/
    /// Instance of use `qdrant_client::qdrant::Filter`
    pub fn search_filter(mut self, search_filter: Filter) -> Self {
        self.search_filter = Some(search_filter);
        self
    }

    /// Build the Store object.
    pub async fn build(mut self) -> Result<Store, Box<dyn Error>> {
        let client = self.client.take().ok_or("'client' is required")?;
        let embedder = self.embedder.take().ok_or("'embedder' is required")?;
        let collection_name = self
            .collection_name
            .take()
            .ok_or("'collection_name' is required")?;

        let collection_exists = client.collection_exists(&collection_name).await?;

        // Delete the collection if it exists and recreate_collection flag is set
        if collection_exists && self.recreate_collection {
            client.delete_collection(&collection_name).await?;
        }

        // Create the collection if it doesn't exist or recreate_collection flag is set
        if !collection_exists || self.recreate_collection {
            // Embed some text to get the dimension of the embeddings
            let embeddings = embedder
                .embed_query("Text to retrieve embeddings dimension")
                .await?;
            let embeddings_dimension = embeddings.len() as u64;

            client
                .create_collection(&CreateCollection {
                    collection_name: collection_name.clone(),
                    vectors_config: Some(VectorsConfig {
                        config: Some(Config::Params(VectorParams {
                            size: embeddings_dimension,
                            distance: Distance::Cosine.into(),
                            ..Default::default()
                        })),
                    }),
                    ..Default::default()
                })
                .await?;
        }

        Ok(Store {
            client,
            embedder,
            collection_name,
            search_filter: self.search_filter,
            content_field: self.content_field,
            metadata_field: self.metadata_field,
        })
    }
}
