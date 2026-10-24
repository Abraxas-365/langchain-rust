use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The `Document` struct represents a document with content, metadata, and a score.
/// The `page_content` field is a string that contains the content of the document.
/// The `metadata` field is a `HashMap` where the keys represent metadata properties and the values represent property values.
/// The `score` field represents a relevance score for the document and is a floating point number.
/// The optional `id` field is a caller-supplied identifier for the final indexed document.
///
/// `Document` is `#[non_exhaustive]`. Downstream crates build it with
/// [`Document::new`] or [`Document::default`] and the `with_*` methods, so the next
/// field can be added without another source break. Adding `id` is still a breaking
/// change for struct literals that already existed. Documents serialized before `id`
/// existed, and the `VectorStore` trait (including the `add_documents!` macro), stay
/// compatible.
///
/// # Usage
/// ```rust,ignore
/// let my_doc = Document::new("This is the document content.".to_string())
///    .with_metadata({
///       let mut metadata = HashMap::new();
///       metadata.insert("author".to_string(), json!("John Doe"));
///       metadata
///   })
///    .with_score(0.75);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Document {
    pub page_content: String,
    pub metadata: HashMap<String, Value>,
    pub score: f64,
    /// Identifier of the final indexed document.
    ///
    /// `None` asks the vector store to generate an id. `Some` is sent to stores that
    /// accept caller-supplied ids so a later insert with the same id replaces the
    /// existing record.
    ///
    /// Ids identify final indexed documents. `TextSplitter::split_documents` builds a
    /// new document per chunk and leaves `id` unset; it does not copy a parent id onto
    /// every chunk. Assign a unique id to each chunk after splitting.
    ///
    /// # Store support
    ///
    /// - **OpenSearch** writes a non-empty id as the bulk action `_id` (index/replace).
    ///   `None` omits `_id` so the server generates one. An empty id is rejected.
    ///   A bulk response is not an atomic multi-document transaction; item errors are
    ///   returned as `Err`.
    /// - **pgvector** stores the id in the existing `uuid` TEXT primary key and upserts
    ///   document, embedding, and metadata when that row already belongs to the same
    ///   collection. The same id in another collection that shares the embedding table
    ///   is an error and rolls the batch back. `None` generates a UUID. An empty id is
    ///   rejected.
    /// - **Qdrant** accepts a UUID string or an unsigned `u64` decimal string and
    ///   upserts that point. `None` generates a UUID. Empty, negative, overflowing, and
    ///   other arbitrary strings are rejected. Repeated ids in one batch keep the last
    ///   document.
    /// - **Chroma** and **Pinecone** use any non-empty id in their upsert, replacing the
    ///   record with that id. `None` generates a UUID.
    /// - **Weaviate** needs a UUID (anything else is rejected before embedding) and
    ///   replaces the object with that id. Ids come back lowercase and hyphenated.
    /// - **Milvus** uses the id as a `VarChar` primary key and upserts the batch.
    ///   Collections with an `autoId` key reject documents that set an id.
    /// - **Redis** uses the id as the key after the prefix and replaces the whole hash
    ///   (the old one is deleted in the same pipeline, so stale fields don't survive).
    ///   `add_documents` returns the full keys; searches return the id without prefix.
    /// - **MongoDB** stores the id as a string `_id` and replaces the document with
    ///   that `_id`. `None` generates an ObjectId.
    ///
    /// Searches in all of the above return the stored id.
    ///
    /// SQLite vec, SQLite VSS, and SurrealDB ignore `id` on insert and keep their
    /// existing id generation. Documents they return leave `id` unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

impl Document {
    /// Constructs a new `Document` with provided `page_content`, an empty `metadata` map, a `score` of 0, and no id.
    pub fn new<S: Into<String>>(page_content: S) -> Self {
        Document {
            page_content: page_content.into(),
            metadata: HashMap::new(),
            score: 0.0,
            id: None,
        }
    }

    /// Sets the `metadata` Map of the `Document` to the provided HashMap.
    pub fn with_metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets the `score` of the `Document` to the provided float.
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }

    /// Sets the caller-supplied id of the final indexed document.
    ///
    /// Give each chunk its own id after splitting; a parent id is not reused for every
    /// chunk. See [`Document::id`] for which stores honor the value.
    ///
    /// - OpenSearch: non-empty bulk `_id` (omitted when this is never called). Empty
    ///   ids are rejected. The index action replaces an existing document.
    /// - pgvector: non-empty TEXT primary key. Upserts inside the same collection;
    ///   the same id in another collection errors and rolls the batch back.
    /// - Qdrant: a UUID or an unsigned `u64` decimal (`0` through `u64::MAX`).
    ///   Empty, negative, overflowing, and other strings are rejected. Upsert
    ///   replaces the point; the last copy in a batch wins.
    /// - Chroma, Pinecone: any non-empty string; upsert replaces the record.
    /// - Weaviate: a UUID; the object with that id is replaced.
    /// - Milvus: any non-empty string with a `VarChar` primary key (upserted).
    ///   `autoId` collections reject it.
    /// - Redis: the key's id part (after the prefix); the old hash is replaced.
    /// - MongoDB: a string `_id`; the document with that `_id` is replaced.
    /// - SQLite vec, SQLite VSS, and SurrealDB ignore this value.
    ///
    /// ```
    /// use langchain_rust::schemas::Document;
    ///
    /// let doc = Document::new("chunk text").with_id("source-42-chunk-0");
    /// assert_eq!(doc.id.as_deref(), Some("source-42-chunk-0"));
    /// ```
    pub fn with_id<S: Into<String>>(mut self, id: S) -> Self {
        self.id = Some(id.into());
        self
    }
}

impl Default for Document {
    /// Provides a default `Document` with an empty `page_content`, an empty `metadata` map, a `score` of 0, and no id.
    fn default() -> Self {
        Document {
            page_content: "".to_string(),
            metadata: HashMap::new(),
            score: 0.0,
            id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn new_and_default_have_no_id_and_with_id_survives_clone() {
        assert!(Document::new("x").id.is_none());
        assert!(Document::default().id.is_none());

        let metadata = HashMap::from([("k".to_string(), json!("v"))]);
        let doc = Document::new("chunk")
            .with_id("source-1")
            .with_metadata(metadata.clone())
            .with_score(0.5);
        let cloned = doc.clone();
        assert_eq!(cloned.id.as_deref(), Some("source-1"));
        assert_eq!(cloned.metadata["k"], json!("v"));
        assert_eq!(cloned.page_content, "chunk");
        assert_eq!(cloned.score, 0.5);
        assert_eq!(doc.metadata, metadata);
    }

    #[test]
    fn serde_round_trip_keeps_id_and_legacy_json_omits_none() {
        let doc = Document::new("chunk")
            .with_id("source-1")
            .with_metadata(HashMap::from([("k".to_string(), json!("v"))]))
            .with_score(0.5);
        let value = serde_json::to_value(&doc).unwrap();
        assert_eq!(value["id"], json!("source-1"));
        assert_eq!(value["metadata"]["k"], json!("v"));
        let restored: Document = serde_json::from_value(value).unwrap();
        assert_eq!(restored.id.as_deref(), Some("source-1"));
        assert_eq!(restored.metadata["k"], json!("v"));
        assert_eq!(restored.page_content, "chunk");
        assert_eq!(restored.score, 0.5);

        let legacy = json!({"page_content": "old", "metadata": {"a": 1}, "score": 0.2});
        let from_legacy: Document = serde_json::from_value(legacy).unwrap();
        assert!(from_legacy.id.is_none());
        assert_eq!(from_legacy.metadata["a"], json!(1));
        assert_eq!(from_legacy.page_content, "old");

        let unset =
            Document::new("old").with_metadata(HashMap::from([("a".to_string(), json!(1))]));
        let serialized = serde_json::to_value(&unset).unwrap();
        assert!(serialized.get("id").is_none());
        assert_eq!(
            serialized,
            json!({"page_content": "old", "metadata": {"a": 1}, "score": 0.0})
        );
    }
}
