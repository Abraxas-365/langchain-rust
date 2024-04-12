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
impl<VS> From<VS> for Box<dyn VectorStore>
where
    VS: 'static + VectorStore,
{
    fn from(vector_store: VS) -> Self {
        Box::new(vector_store)
    }
}

#[macro_export]
macro_rules! add_documents {
    ($obj:expr, $docs:expr) => {
        $obj.add_documents($docs, &$crate::vectorstore::VecStoreOptions::default())
    };
    ($obj:expr, $docs:expr, $opt:expr) => {
        $obj.add_documents($docs, $opt)
    };
}

#[macro_export]
macro_rules! similarity_search {
    ($obj:expr, $query:expr, $limit:expr) => {
        $obj.similarity_search(
            $query,
            $limit,
            &$crate::vectorstore::VecStoreOptions::default(),
        )
    };
    ($obj:expr, $query:expr, $limit:expr, $opt:expr) => {
        $obj.similarity_search($query, $limit, $opt)
    };
}

// Retriever is a retriever for vector stores.
pub struct Retriever {
    vstore: Box<dyn VectorStore>,
    num_docs: usize,
    options: VecStoreOptions,
}
impl Retriever {
    pub fn new<V: Into<Box<dyn VectorStore>>>(vstore: V, num_docs: usize) -> Self {
        Retriever {
            vstore: vstore.into(),
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
impl schemas::Retriever for Retriever {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>, Box<dyn Error>> {
        self.vstore
            .similarity_search(query, self.num_docs, &self.options)
            .await
    }
}
