use std::error::Error;

use async_trait::async_trait;

use crate::schemas::{self, Document};

use super::VecStoreOptions;

// VectorStore is the trait for saving and querying documents in the
// form of vector embeddings.
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &VecStoreOptions,
    ) -> Result<Vec<String>, Box<dyn Error>>;
    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &VecStoreOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>>;
}

// Retriever is a retriever for vector stores.
pub struct Retriver {
    vstore: Box<dyn VectorStore>,
    num_docs: usize,
    options: VecStoreOptions,
}
impl Retriver {
    pub fn new<V: VectorStore + 'static>(vstore: V, num_docs: usize) -> Self {
        Retriver {
            vstore: Box::new(vstore),
            num_docs,
            options: VecStoreOptions::default(),
        }
    }

    pub fn with_options(mut self, options: VecStoreOptions) -> Self {
        self.options = options;
        self
    }
}

#[async_trait]
impl schemas::Retriever for Retriver {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>, Box<dyn Error>> {
        self.vstore
            .similarity_search(query, self.num_docs, &self.options)
            .await
    }
}
