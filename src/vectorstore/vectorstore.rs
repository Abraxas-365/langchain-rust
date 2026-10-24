use std::error::Error;

use async_trait::async_trait;

use crate::schemas::{self, Document};

use super::VecStoreOptions;

// VectorStore is the trait for saving and querying documents in the
// form of vector embeddings.
#[async_trait]
pub trait VectorStore: Send + Sync {
    type Options;

    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &Self::Options,
    ) -> Result<Vec<String>, Box<dyn Error>>;

    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &Self::Options,
    ) -> Result<Vec<Document>, Box<dyn Error>>;
}

/// The id each document is written with: its own [`Document::id`] when set, otherwise
/// one from `generate`. Fails on an empty id before anything is sent, so a bad id can't
/// look like a successful write.
#[allow(dead_code)] // Unused when no store using it is enabled.
pub(crate) fn document_ids(
    docs: &[Document],
    mut generate: impl FnMut() -> String,
) -> Result<Vec<String>, Box<dyn Error>> {
    docs.iter()
        .map(|doc| match doc.id.as_deref() {
            Some("") => Err("document id must not be empty".into()),
            Some(id) => Ok(id.to_string()),
            None => Ok(generate()),
        })
        .collect()
}

impl<VS, F> From<VS> for Box<dyn VectorStore<Options = F>>
where
    VS: 'static + VectorStore<Options = F>,
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
pub struct Retriever<F> {
    vstore: Box<dyn VectorStore<Options = VecStoreOptions<F>>>,
    num_docs: usize,
    options: VecStoreOptions<F>,
}
impl<F> Retriever<F> {
    pub fn new<V: Into<Box<dyn VectorStore<Options = VecStoreOptions<F>>>>>(
        vstore: V,
        num_docs: usize,
    ) -> Self {
        Retriever {
            vstore: vstore.into(),
            num_docs,
            options: VecStoreOptions::<F>::new(),
        }
    }

    pub fn with_options(mut self, options: VecStoreOptions<F>) -> Self {
        self.options = options;
        self
    }
}

#[async_trait]
impl<O: Sync + Send> schemas::Retriever for Retriever<O> {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>, Box<dyn Error>> {
        self.vstore
            .similarity_search(query, self.num_docs, &self.options)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_ids_keep_caller_ids_and_generate_the_rest_in_order() {
        let mut next = 0;
        let ids = document_ids(
            &[
                Document::new("a"),
                Document::new("b").with_id("doc-b"),
                Document::new("c"),
            ],
            || {
                next += 1;
                format!("gen-{next}")
            },
        )
        .unwrap();
        assert_eq!(ids, ["gen-1", "doc-b", "gen-2"]);
        assert!(document_ids(&[], || unreachable!()).unwrap().is_empty());
    }

    #[test]
    fn document_ids_reject_empty_ids() {
        let err = document_ids(
            &[Document::new("a"), Document::new("b").with_id("")],
            || "gen".into(),
        )
        .unwrap_err();
        assert_eq!(err.to_string(), "document id must not be empty");
    }
}
