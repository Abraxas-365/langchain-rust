use async_trait::async_trait;
use qdrant_client::qdrant::{Filter, PointStruct, SearchPoints};
use serde_json::json;
use std::error::Error;
use std::sync::Arc;

pub use qdrant_client::client::QdrantClient;

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{VecStoreOptions, VectorStore},
};
use uuid::Uuid;

pub struct Store {
    pub client: QdrantClient,
    pub embedder: Arc<dyn Embedder>,
    pub collection_name: String,
    pub content_field: String,
    pub metadata_field: String,
    pub search_filter: Option<Filter>,
}

#[async_trait]
impl VectorStore for Store {
    /// Add documents to the store.
    /// Returns a list of document IDs added to the Qdrant collection.
    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &VecStoreOptions,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let texts: Vec<String> = docs.iter().map(|d| d.page_content.clone()).collect();

        let ids = docs.iter().map(|_| Uuid::new_v4().to_string());
        let vectors = embedder.embed_documents(&texts).await?.into_iter();
        let payloads = docs.iter().map(|d| {
            json!({
                &self.content_field: d.page_content,
                &self.metadata_field: d.metadata,
            })
        });

        let mut points: Vec<PointStruct> = Vec::with_capacity(docs.len());

        for (id, (vector, payload)) in ids.clone().zip(vectors.zip(payloads)) {
            let vector: Vec<f32> = vector.into_iter().map(|f| f as f32).collect();
            let point = PointStruct::new(id, vector, payload.try_into().unwrap());
            points.push(point);
        }

        self.client
            .upsert_points_blocking(self.collection_name.clone(), None, points, None)
            .await?;

        Ok(ids.collect())
    }

    /// Perform a similarity search on the store.
    /// Returns a list of documents similar to the query.
    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &VecStoreOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        if opt.name_space.is_some() {
            return Err("Qdrant doesn't support namespaces".into());
        }

        if opt.filters.is_some() {
            return Err(
                "'qdrant_client' doesn't support 'serde_json::Value' filters. 
            Use `search_filter` when constructing VectorStore instead"
                    .into(),
            );
        }

        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let query_vector: Vec<f32> = embedder
            .embed_query(query)
            .await?
            .into_iter()
            .map(|f| f as f32)
            .collect();

        let results = self
            .client
            .search_points(&SearchPoints {
                collection_name: self.collection_name.clone(),
                vector: query_vector,
                limit: limit as u64,
                with_payload: Some(true.into()),
                score_threshold: opt.score_threshold,
                filter: self.search_filter.clone(),
                ..Default::default()
            })
            .await?;

        let documents = results
            .result
            .into_iter()
            .map(|scored_point| {
                let payload = scored_point.payload;

                let page_content = payload[&self.content_field].to_string();
                let metadata =
                    serde_json::from_value(payload[&self.metadata_field].clone().into_json())
                        .unwrap();
                let score = scored_point.score as f64;
                Document {
                    page_content,
                    metadata,
                    score,
                }
            })
            .collect();

        Ok(documents)
    }
}
