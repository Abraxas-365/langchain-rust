use std::{collections::HashMap, error::Error, future::Future, sync::Arc, time::Duration};

use async_trait::async_trait;
use futures::TryStreamExt;
use mongodb::{
    bson::{doc, oid::ObjectId, Bson, Document as BsonDocument},
    Collection, SearchIndexModel, SearchIndexType,
};
use serde_json::Value;
use tokio::sync::OnceCell;

use crate::{
    embedding::Embedder,
    schemas::Document,
    vectorstore::{VecStoreOptions, VectorStore},
};

/// Projected `vectorSearchScore`. Not a field anyone stores.
pub(super) const SCORE_FIELD: &str = "__vector_score";
/// Atlas caps `numCandidates` (and `limit`) at 10,000.
const MAX_CANDIDATES: usize = 10_000;

/// The similarity function of a new index. Atlas normalizes every score to `0..=1`,
/// higher is closer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MongoSimilarity {
    #[default]
    Cosine,
    Euclidean,
    DotProduct,
}

impl MongoSimilarity {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Cosine => "cosine",
            Self::Euclidean => "euclidean",
            Self::DotProduct => "dotProduct",
        }
    }
}

/// A MongoDB Atlas vector store. See the [module docs](super).
pub struct Store {
    pub(super) collection: Collection<BsonDocument>,
    pub(super) embedder: Arc<dyn Embedder>,
    pub(super) index_name: String,
    pub(super) text_key: String,
    pub(super) embedding_key: String,
    pub(super) metadata_key: String,
    pub(super) namespace_key: String,
    pub(super) namespace: Option<String>,
    pub(super) filter_paths: Vec<String>,
    pub(super) similarity: MongoSimilarity,
    pub(super) num_candidates: usize,
    pub(super) create_index: bool,
    pub(super) index_timeout: Duration,
    pub(super) index_ready: OnceCell<()>,
}

/// Polls `status` (`None`: index missing, `Some(queryable)`) until the index is queryable.
pub(super) async fn wait_until_queryable<F, Fut>(
    mut status: F,
    interval: Duration,
    timeout: Duration,
) -> Result<(), Box<dyn Error + Send + Sync>>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Option<bool>, Box<dyn Error + Send + Sync>>>,
{
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        match status().await? {
            Some(true) => return Ok(()),
            Some(false) => {}
            None => return Err("the vector search index disappeared while building".into()),
        }
        if tokio::time::Instant::now() + interval > deadline {
            return Err(format!(
                "the vector search index was not queryable after {timeout:?}; \
                 raise index_timeout or check the index in Atlas"
            )
            .into());
        }
        tokio::time::sleep(interval).await;
    }
}

/// `numCandidates` for a query: `limit × multiplier`, at least `limit`, at most 10,000.
pub(super) fn num_candidates(limit: usize, multiplier: usize) -> usize {
    limit
        .saturating_mul(multiplier)
        .clamp(limit, MAX_CANDIDATES.max(limit))
}

fn bson_to_json(value: Bson) -> Value {
    match value {
        // Relaxed extended JSON turns these into `{"$oid": ..}` / `{"$date": ..}`;
        // plain strings are friendlier in metadata.
        Bson::ObjectId(id) => Value::String(id.to_hex()),
        Bson::DateTime(dt) => dt
            .try_to_rfc3339_string()
            .map(Value::String)
            .unwrap_or_else(|_| Bson::DateTime(dt).into_relaxed_extjson()),
        Bson::Document(doc) => {
            Value::Object(doc.into_iter().map(|(k, v)| (k, bson_to_json(v))).collect())
        }
        Bson::Array(items) => Value::Array(items.into_iter().map(bson_to_json).collect()),
        other => other.into_relaxed_extjson(),
    }
}

impl Store {
    pub fn collection(&self) -> &Collection<BsonDocument> {
        &self.collection
    }

    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// The `vectorSearch` index definition for `dim`-dimensional embeddings.
    pub(super) fn index_definition(&self, dim: usize) -> BsonDocument {
        let mut fields = vec![
            Bson::Document(doc! {
                "type": "vector",
                "path": &self.embedding_key,
                "numDimensions": dim as i64,
                "similarity": self.similarity.as_str(),
            }),
            Bson::Document(doc! { "type": "filter", "path": &self.namespace_key }),
        ];
        for path in &self.filter_paths {
            fields.push(Bson::Document(doc! { "type": "filter", "path": path }));
        }
        doc! { "fields": fields }
    }

    /// The stored form of a document.
    pub(super) fn to_bson(
        &self,
        doc: &Document,
        vector: &[f64],
        namespace: Option<&str>,
    ) -> Result<BsonDocument, Box<dyn Error>> {
        let mut metadata = BsonDocument::new();
        let mut keys: Vec<_> = doc.metadata.keys().collect();
        keys.sort();
        for key in keys {
            let value = Bson::try_from(doc.metadata[key].clone())
                .map_err(|e| format!("metadata {key:?} is not valid BSON: {e}"))?;
            metadata.insert(key.clone(), value);
        }
        let mut stored = doc! {
            &self.text_key: &doc.page_content,
            &self.embedding_key: vector.iter().map(|x| Bson::Double(*x)).collect::<Vec<_>>(),
        };
        if let Some(ns) = namespace {
            stored.insert(&self.namespace_key, ns);
        }
        stored.insert(&self.metadata_key, metadata);
        Ok(stored)
    }

    /// The `$vectorSearch` pre-filter: namespace equality AND the user's MQL filter.
    pub(super) fn filter(
        &self,
        namespace: Option<&str>,
        filters: Option<&Value>,
    ) -> Result<Option<BsonDocument>, Box<dyn Error>> {
        let mut clauses = Vec::new();
        if let Some(ns) = namespace {
            clauses.push(doc! { &self.namespace_key: { "$eq": ns } });
        }
        match filters {
            None | Some(Value::Null) => {}
            Some(Value::Object(fields)) if fields.is_empty() => {}
            Some(value @ Value::Object(_)) => match Bson::try_from(value.clone()) {
                Ok(Bson::Document(doc)) => clauses.push(doc),
                Ok(_) => unreachable!("a JSON object converts to a BSON document"),
                Err(e) => return Err(format!("invalid MongoDB filter: {e}").into()),
            },
            Some(other) => {
                return Err(format!(
                    "MongoDB filters are an MQL object like {{\"metadata.year\": {{\"$gte\": 2020}}}}, got {other}"
                )
                .into())
            }
        }
        Ok(match clauses.len() {
            0 => None,
            1 => clauses.pop(),
            _ => Some(doc! { "$and": clauses }),
        })
    }

    pub(super) fn pipeline(
        &self,
        vector: &[f64],
        limit: usize,
        filter: Option<BsonDocument>,
    ) -> Vec<BsonDocument> {
        let mut search = doc! {
            "index": &self.index_name,
            "path": &self.embedding_key,
            "queryVector": vector.iter().map(|x| Bson::Double(*x)).collect::<Vec<_>>(),
            "numCandidates": num_candidates(limit, self.num_candidates) as i64,
            "limit": limit as i64,
        };
        if let Some(filter) = filter {
            search.insert("filter", filter);
        }
        vec![
            doc! { "$vectorSearch": search },
            doc! { "$project": {
                &self.embedding_key: 0,
                SCORE_FIELD: { "$meta": "vectorSearchScore" },
            } },
        ]
    }

    pub(super) fn to_document(&self, mut stored: BsonDocument) -> Result<Document, Box<dyn Error>> {
        let score = match stored.remove(SCORE_FIELD) {
            Some(Bson::Double(s)) => s,
            Some(Bson::Int32(s)) => s.into(),
            Some(Bson::Int64(s)) => s as f64,
            other => return Err(format!("search result without a score: {other:?}").into()),
        };
        let page_content = match stored.remove(&self.text_key) {
            Some(Bson::String(text)) => text,
            Some(other) => {
                return Err(format!("{:?} is not a string: {other}", self.text_key).into())
            }
            None => String::new(),
        };
        let id = match stored.remove("_id") {
            Some(Bson::ObjectId(id)) => Some(id.to_hex()),
            Some(Bson::String(id)) => Some(id),
            Some(Bson::Null) | None => None,
            Some(other) => Some(other.to_string()),
        };
        let metadata: HashMap<String, Value> = match stored.remove(&self.metadata_key) {
            Some(Bson::Document(metadata)) => metadata
                .into_iter()
                .map(|(k, v)| (k, bson_to_json(v)))
                .collect(),
            _ => HashMap::new(),
        };
        Ok(Document {
            page_content,
            metadata,
            score,
            id,
        })
    }

    /// The index's state: `None` when missing, else whether it can be queried yet.
    async fn index_status(&self) -> Result<Option<bool>, Box<dyn Error + Send + Sync>> {
        let mut cursor = self
            .collection
            .list_search_indexes()
            .name(self.index_name.clone())
            .await?;
        Ok(cursor
            .try_next()
            .await?
            .map(|index| index.get_bool("queryable").unwrap_or(false)))
    }

    /// Creates the index (after the first insert, so the collection exists) and waits
    /// until it is queryable. Runs once per store.
    async fn ensure_index(&self, dim: usize) -> Result<(), Box<dyn Error + Send + Sync>> {
        if !self.create_index {
            return Ok(());
        }
        self.index_ready
            .get_or_try_init(|| async {
                if self.index_status().await? == Some(true) {
                    return Ok(());
                }
                let model = SearchIndexModel::builder()
                    .name(self.index_name.clone())
                    .index_type(SearchIndexType::VectorSearch)
                    .definition(self.index_definition(dim))
                    .build();
                match self.collection.create_search_index(model).await {
                    Ok(_) => {}
                    // Already there with another definition (68 IndexAlreadyExists), or
                    // created concurrently (Atlas reports that as BadValue "duplicate index
                    // names"). Either way, use the index if it exists now.
                    Err(err) => {
                        if self.index_status().await?.is_none() {
                            return Err(err.into());
                        }
                    }
                }
                wait_until_queryable(
                    || self.index_status(),
                    Duration::from_millis(500),
                    self.index_timeout,
                )
                .await
            })
            .await
            .map(|_| ())
    }
}

#[async_trait]
impl VectorStore for Store {
    type Options = VecStoreOptions<Value>;

    /// Embeds and stores the documents, returning their `_id`s (hex for generated
    /// ObjectIds).
    ///
    /// A [`Document::id`] is stored as a string `_id` and replaces the document with
    /// that `_id` (`replace_one` with upsert). Documents without one are inserted
    /// together and get an ObjectId.
    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &Self::Options,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }
        // `_id`s are fixed before inserting so the result order is the input order.
        let ids: Vec<Bson> = docs
            .iter()
            .map(|doc| match doc.id.as_deref() {
                Some("") => Err("document id must not be empty"),
                Some(id) => Ok(Bson::String(id.to_string())),
                None => Ok(Bson::ObjectId(ObjectId::new())),
            })
            .collect::<Result<_, _>>()?;
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let texts: Vec<String> = docs.iter().map(|d| d.page_content.clone()).collect();
        let vectors = embedder.embed_documents(&texts).await?;
        if vectors.len() != docs.len() {
            return Err(format!(
                "embedder returned {} vectors for {} documents",
                vectors.len(),
                docs.len()
            )
            .into());
        }
        let namespace = opt.name_space.as_deref().or(self.namespace.as_deref());
        let stored = docs
            .iter()
            .zip(&vectors)
            .zip(&ids)
            .map(|((doc, vector), id)| {
                let mut stored = self.to_bson(doc, vector, namespace)?;
                stored.insert("_id", id.clone());
                Ok(stored)
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

        let (replaced, inserted): (Vec<_>, Vec<_>) = stored
            .into_iter()
            .partition(|stored| matches!(stored.get("_id"), Some(Bson::String(_))));
        if !inserted.is_empty() {
            self.collection.insert_many(inserted).await?;
        }
        for stored in replaced {
            let filter = doc! { "_id": stored.get("_id").cloned().unwrap_or(Bson::Null) };
            self.collection
                .replace_one(filter, stored)
                .upsert(true)
                .await?;
        }

        self.ensure_index(vectors[0].len())
            .await
            .map_err(|e| -> Box<dyn Error> { e.to_string().into() })?;
        Ok(ids
            .into_iter()
            .map(|id| match id {
                Bson::ObjectId(id) => id.to_hex(),
                Bson::String(id) => id,
                other => other.to_string(),
            })
            .collect())
    }

    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &Self::Options,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let namespace = opt.name_space.as_deref().or(self.namespace.as_deref());
        let filter = self.filter(namespace, opt.filters.as_ref())?;
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let vector = embedder.embed_query(query).await?;

        let results: Vec<BsonDocument> = self
            .collection
            .aggregate(self.pipeline(&vector, limit, filter))
            .await?
            .try_collect()
            .await?;
        let mut docs = Vec::with_capacity(results.len());
        for result in results {
            let doc = self.to_document(result)?;
            if opt
                .score_threshold
                .is_some_and(|threshold| doc.score < f64::from(threshold))
            {
                continue;
            }
            docs.push(doc);
        }
        Ok(docs)
    }
}
