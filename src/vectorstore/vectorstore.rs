use std::error::Error;

use async_trait::async_trait;

use crate::schemas::{self, Document};

use super::VecStoreOptions;

// VectorStore is the trait for saving and querying documents in the
// form of vector embeddings.
#[async_trait]
pub trait VectorStore: Send + Sync {
    type Options: Sync;

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

    /// Deletes the documents with these ids (the ones [`add_documents`] returned). Ids that
    /// don't exist are ignored. Stores that can't delete by id return an error.
    ///
    /// [`add_documents`]: VectorStore::add_documents
    async fn delete(&self, ids: &[String], opt: &Self::Options) -> Result<(), Box<dyn Error>> {
        let _ = (ids, opt);
        Err(unsupported::<Self>("delete"))
    }

    /// Fetches documents by id, in the order asked for; ids that don't exist are skipped.
    /// Each document has its [`Document::id`] set and no score. Stores that can't look
    /// documents up by id return an error.
    async fn get_by_ids(
        &self,
        ids: &[String],
        opt: &Self::Options,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let _ = (ids, opt);
        Err(unsupported::<Self>("get_by_ids"))
    }
}

fn unsupported<T: ?Sized>(method: &str) -> Box<dyn Error> {
    let store = std::any::type_name::<T>();
    let store = store.split('<').next().unwrap_or(store);
    let store = store.rsplit("::").next().unwrap_or(store);
    format!("{store} does not support {method}").into()
}

/// Puts documents fetched in any order back in the order of `ids`, dropping ids that
/// weren't found (and duplicates of the same id).
#[allow(dead_code)] // Unused when no store using it is enabled.
pub(crate) fn in_id_order(ids: &[String], docs: Vec<Document>) -> Vec<Document> {
    let mut by_id: std::collections::HashMap<String, Document> = docs
        .into_iter()
        .filter_map(|doc| Some((doc.id.clone()?, doc)))
        .collect();
    ids.iter().filter_map(|id| by_id.remove(id)).collect()
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

    struct SearchOnly;

    #[async_trait]
    impl VectorStore for SearchOnly {
        type Options = VecStoreOptions<serde_json::Value>;
        async fn add_documents(
            &self,
            _: &[Document],
            _: &Self::Options,
        ) -> Result<Vec<String>, Box<dyn Error>> {
            Ok(vec![])
        }
        async fn similarity_search(
            &self,
            _: &str,
            _: usize,
            _: &Self::Options,
        ) -> Result<Vec<Document>, Box<dyn Error>> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn delete_and_get_default_to_unsupported() {
        let ids = ["a".to_string()];
        let opt = VecStoreOptions::default();
        let err = SearchOnly.delete(&ids, &opt).await.unwrap_err();
        assert_eq!(err.to_string(), "SearchOnly does not support delete");
        let err = SearchOnly.get_by_ids(&ids, &opt).await.unwrap_err();
        assert_eq!(err.to_string(), "SearchOnly does not support get_by_ids");
        // Also through a trait object.
        let boxed: Box<dyn VectorStore<Options = VecStoreOptions<serde_json::Value>>> =
            Box::new(SearchOnly);
        assert!(boxed.delete(&ids, &opt).await.is_err());
    }

    #[test]
    fn in_id_order_follows_the_request_and_skips_missing() {
        let doc = |id: &str| Document::new(format!("text {id}")).with_id(id);
        let ids: Vec<String> = ["c", "missing", "a", "c"].map(String::from).to_vec();
        let out = in_id_order(&ids, vec![doc("a"), doc("b"), doc("c")]);
        let got: Vec<_> = out.iter().map(|d| d.id.as_deref().unwrap()).collect();
        assert_eq!(got, ["c", "a"]);
        assert_eq!(out[0].page_content, "text c");
    }
}
