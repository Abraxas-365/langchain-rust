use std::{collections::HashMap, error::Error, sync::Arc, time::Duration};

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{json, Map, Value};
use tokio::sync::OnceCell;

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{VecStoreOptions, VectorStore},
};

/// Rows per insert request.
pub(super) const BATCH_SIZE: usize = 100;
/// Name of the filter template parameter holding the namespace.
pub(super) const NAMESPACE_PARAM: &str = "__lc_namespace";
/// Milvus "collection not found" codes (search answers 100, describe/load-state too).
const COLLECTION_NOT_FOUND: i64 = 100;
const LOAD_POLL_INTERVAL: Duration = Duration::from_millis(200);

/// Filters are Milvus boolean expressions, either as a string, e.g.
/// `json!("metadata[\"source\"] == \"a.txt\"")`, or as a template with its parameters,
/// `json!({ "expr": "metadata[\"year\"] >= {min}", "params": { "min": 2020 } })`.
pub type MilvusOptions = VecStoreOptions<Value>;

/// [Milvus](https://milvus.io/) (and Zilliz Cloud) vector store over the REST API v2
/// (Milvus 2.4+). Build it with [`StoreBuilder`](super::StoreBuilder).
///
/// - The first write creates the collection if it's missing: a `VarChar` primary key
///   (UUIDs), the text, a `FloatVector` with the embedder's dimension, a `JSON` metadata
///   field and a `VarChar` namespace used as the partition key, so per-tenant searches only
///   touch that tenant's data. It gets an `AUTOINDEX` index with the configured metric and
///   the configured consistency level (default `Strong`, so writes are searchable at once).
/// - Existing collections work too: the schema is read once. A primary key with `autoId`
///   gets Milvus' ids, a `VarChar` one gets UUIDs. The namespace and metadata fields are
///   optional; other scalar fields come back as metadata. A collection that isn't loaded
///   is loaded on first use.
/// - Options:
///   - `filters`: a Milvus boolean expression (see [`MilvusOptions`]).
///   - `name_space`: stored in the namespace field on writes and required to match on
///     searches. It falls back to the builder's `namespace`.
///   - `score_threshold`: keeps `score >= threshold`.
///   - `embedder`: per-call override.
/// - `Document::score` is the similarity for `COSINE`/`IP` and `-distance` for `L2`, so
///   higher is always better.
pub struct Store {
    pub(super) client: Client,
    pub(super) url: String,
    pub(super) token: Option<String>,
    pub(super) db_name: Option<String>,
    pub(super) collection: String,
    pub(super) embedder: Arc<dyn Embedder>,
    pub(super) primary_field: String,
    pub(super) text_field: String,
    pub(super) vector_field: String,
    pub(super) metadata_field: String,
    pub(super) namespace_field: String,
    pub(super) namespace: Option<String>,
    pub(super) metric: String,
    pub(super) consistency_level: String,
    pub(super) max_text_length: usize,
    pub(super) create_collection: bool,
    pub(super) load_timeout: Duration,
    pub(super) schema: OnceCell<Schema>,
}

/// What the store needs to know about the collection.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Schema {
    pub primary: String,
    pub auto_id: bool,
    pub has_namespace: bool,
    pub has_metadata: bool,
    /// Everything but vectors and the primary key.
    pub output_fields: Vec<String>,
    pub metric: String,
}

/// A search `filter` expression and its `exprParams`.
pub(super) type Filter = (Option<String>, Map<String, Value>);

/// Milvus answers HTTP 200 with `{"code": N, "message": ...}`; anything but 0 is an error.
pub(super) fn milvus_result(response: Value) -> Result<Value, MilvusError> {
    match response["code"].as_i64() {
        Some(0) => Ok(response),
        code => Err(MilvusError {
            code: code.unwrap_or(-1),
            message: response["message"]
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| response.to_string()),
        }),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MilvusError {
    pub code: i64,
    pub message: String,
}

impl std::fmt::Display for MilvusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Milvus error {}: {}", self.code, self.message)
    }
}

impl Error for MilvusError {}

/// Reads [`Schema`] from a `collections/describe` answer.
pub(super) fn parse_schema(
    describe: &Value,
    text_field: &str,
    vector_field: &str,
    metadata_field: &str,
    namespace_field: &str,
) -> Result<Schema, Box<dyn Error>> {
    let fields = describe["fields"]
        .as_array()
        .ok_or("Milvus describe answer has no fields")?;
    let field = |name: &str| fields.iter().find(|f| f["name"] == json!(name));
    let primary = fields
        .iter()
        .find(|f| f["primaryKey"] == json!(true))
        .ok_or("Milvus collection has no primary key")?;
    let auto_id = primary["autoId"] == json!(true) || describe["autoId"] == json!(true);
    if !auto_id && primary["type"] != json!("VarChar") {
        return Err(format!(
            "Milvus primary key {} is {} without autoId; use a VarChar or an autoId key",
            primary["name"], primary["type"]
        )
        .into());
    }
    match field(text_field) {
        Some(f) if f["type"] == json!("VarChar") => {}
        _ => return Err(format!("Milvus collection has no VarChar field {text_field:?}").into()),
    }
    match field(vector_field) {
        Some(f) if f["type"] == json!("FloatVector") => {}
        _ => {
            return Err(
                format!("Milvus collection has no FloatVector field {vector_field:?}").into(),
            )
        }
    }
    let primary_name = primary["name"].as_str().unwrap_or_default().to_string();
    // The namespace is internal; vectors are big. The primary key comes back anyway.
    let output_fields = fields
        .iter()
        .filter(|f| {
            f["name"] != json!(primary_name)
                && f["name"] != json!(namespace_field)
                && !f["type"].as_str().unwrap_or_default().ends_with("Vector")
        })
        .filter_map(|f| f["name"].as_str().map(str::to_string))
        .collect();
    let metric = describe["indexes"]
        .as_array()
        .and_then(|indexes| {
            indexes
                .iter()
                .find(|i| i["fieldName"] == json!(vector_field))
        })
        .and_then(|i| i["metricType"].as_str())
        .unwrap_or("COSINE")
        .to_ascii_uppercase();
    Ok(Schema {
        primary: primary_name,
        auto_id,
        has_namespace: field(namespace_field).is_some(),
        has_metadata: field(metadata_field).is_some_and(|f| f["type"] == json!("JSON")),
        output_fields,
        metric,
    })
}

/// Higher is better for every metric.
pub(super) fn score(metric: &str, distance: f64) -> f64 {
    match metric {
        "COSINE" | "IP" => distance,
        _ => -distance,
    }
}

impl Store {
    fn body(&self, mut body: Value) -> Value {
        body["collectionName"] = json!(self.collection);
        if let Some(db) = &self.db_name {
            body["dbName"] = json!(db);
        }
        body
    }

    pub(super) async fn post(&self, path: &str, body: Value) -> Result<Value, Box<dyn Error>> {
        let mut request = self
            .client
            .post(format!("{}/v2/vectordb/{path}", self.url))
            .json(&self.body(body));
        if let Some(token) = &self.token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await?;
        let status = response.status();
        let text = response.text().await?;
        if !status.is_success() {
            return Err(format!("Milvus HTTP error ({status}): {text}").into());
        }
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| format!("invalid Milvus answer ({e}): {text}"))?;
        Ok(milvus_result(value)?)
    }

    /// The `collections/create` body for a new collection.
    pub(super) fn create_body(&self, dim: usize) -> Value {
        json!({
            "schema": {
                "autoId": false,
                "enableDynamicField": false,
                "fields": [
                    { "fieldName": self.primary_field, "dataType": "VarChar", "isPrimary": true,
                      "elementTypeParams": { "max_length": 64 } },
                    { "fieldName": self.text_field, "dataType": "VarChar",
                      "elementTypeParams": { "max_length": self.max_text_length } },
                    { "fieldName": self.vector_field, "dataType": "FloatVector",
                      "elementTypeParams": { "dim": dim } },
                    { "fieldName": self.metadata_field, "dataType": "JSON" },
                    { "fieldName": self.namespace_field, "dataType": "VarChar",
                      "isPartitionKey": true, "elementTypeParams": { "max_length": 256 } },
                ],
            },
            "indexParams": [{
                "fieldName": self.vector_field,
                "indexName": self.vector_field,
                "metricType": self.metric,
                "indexType": "AUTOINDEX",
            }],
            "params": { "consistencyLevel": self.consistency_level },
        })
    }

    async fn describe(&self) -> Result<Option<Value>, Box<dyn Error>> {
        match self.post("collections/describe", json!({})).await {
            Ok(answer) => Ok(Some(answer["data"].clone())),
            Err(e) => match e.downcast_ref::<MilvusError>() {
                Some(e) if e.code == COLLECTION_NOT_FOUND => Ok(None),
                _ => Err(e),
            },
        }
    }

    async fn ensure_loaded(&self, describe: &Value) -> Result<(), Box<dyn Error>> {
        if describe["load"] == json!("LoadStateLoaded") {
            return Ok(());
        }
        self.post("collections/load", json!({})).await?;
        let deadline = tokio::time::Instant::now() + self.load_timeout;
        loop {
            let state = self.post("collections/get_load_state", json!({})).await?;
            if state["data"]["loadState"] == json!("LoadStateLoaded") {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(format!(
                    "Milvus collection {} wasn't loaded after {:?} ({})",
                    self.collection, self.load_timeout, state["data"]
                )
                .into());
            }
            tokio::time::sleep(LOAD_POLL_INTERVAL).await;
        }
    }

    /// The collection's schema, creating the collection when `create_dim` is set. `None`
    /// if it doesn't exist (and wasn't created). Cached once found.
    pub(super) async fn schema(
        &self,
        create_dim: Option<usize>,
    ) -> Result<Option<&Schema>, Box<dyn Error>> {
        if let Some(schema) = self.schema.get() {
            return Ok(Some(schema));
        }
        // Bound first: a `?` temporary in the match scrutinee would live across the awaits
        // below, and `Box<dyn Error>` isn't Send.
        let existing = self.describe().await?;
        let describe = match (existing, create_dim) {
            (Some(describe), _) => describe,
            (None, Some(dim)) if self.create_collection => {
                // Creating the same collection twice with the same parameters is fine in
                // Milvus, so concurrent first writes don't race.
                self.post("collections/create", self.create_body(dim))
                    .await?;
                self.describe()
                    .await?
                    .ok_or("Milvus collection disappeared right after creation")?
            }
            (None, Some(_)) => {
                return Err(format!(
                    "Milvus collection {} doesn't exist and create_collection is off",
                    self.collection
                )
                .into())
            }
            (None, None) => return Ok(None),
        };
        let schema = parse_schema(
            &describe,
            &self.text_field,
            &self.vector_field,
            &self.metadata_field,
            &self.namespace_field,
        )?;
        self.ensure_loaded(&describe).await?;
        let _ = self.schema.set(schema);
        Ok(self.schema.get())
    }

    fn namespace<'a>(&'a self, opt: &'a MilvusOptions) -> Option<&'a str> {
        opt.name_space.as_deref().or(self.namespace.as_deref())
    }

    /// The `filter` expression and its `exprParams`.
    pub(super) fn filter(
        &self,
        opt: &MilvusOptions,
        schema: &Schema,
    ) -> Result<Filter, Box<dyn Error>> {
        let (user, mut params) = match &opt.filters {
            None => (None, Map::new()),
            Some(Value::String(expr)) => (Some(expr.clone()), Map::new()),
            Some(Value::Object(filter)) => {
                let expr = filter
                    .get("expr")
                    .and_then(Value::as_str)
                    .ok_or("Milvus filter objects need an `expr` string")?;
                let params = match filter.get("params") {
                    None | Some(Value::Null) => Map::new(),
                    Some(Value::Object(params)) => params.clone(),
                    Some(_) => return Err("Milvus filter `params` must be an object".into()),
                };
                if let Some(extra) = filter.keys().find(|k| *k != "expr" && *k != "params") {
                    return Err(format!("unknown Milvus filter key {extra:?}").into());
                }
                (Some(expr.to_string()), params)
            }
            Some(other) => {
                return Err(format!(
                "Milvus filters are an expression string or {{\"expr\", \"params\"}}, got {other}"
            )
                .into())
            }
        };
        let namespace = match self.namespace(opt) {
            None => None,
            Some(_) if !schema.has_namespace => {
                return Err(format!(
                    "Milvus collection {} has no {:?} field for namespaces",
                    self.collection, self.namespace_field
                )
                .into())
            }
            Some(ns) => {
                if params.contains_key(NAMESPACE_PARAM) {
                    return Err(format!("{NAMESPACE_PARAM} is a reserved filter parameter").into());
                }
                params.insert(NAMESPACE_PARAM.to_string(), json!(ns));
                Some(format!("{} == {{{NAMESPACE_PARAM}}}", self.namespace_field))
            }
        };
        let expr = match (namespace, user) {
            (Some(ns), Some(user)) => Some(format!("{ns} and ({user})")),
            (ns, user) => ns.or(user),
        };
        Ok((expr, params))
    }

    pub(super) fn row(
        &self,
        doc: &Document,
        vector: &[f64],
        id: Option<&str>,
        namespace: Option<&str>,
        schema: &Schema,
    ) -> Result<Value, Box<dyn Error>> {
        let mut row = Map::new();
        if let Some(id) = id {
            row.insert(schema.primary.clone(), json!(id));
        }
        row.insert(self.text_field.clone(), json!(doc.page_content));
        row.insert(self.vector_field.clone(), json!(vector));
        if schema.has_metadata {
            row.insert(self.metadata_field.clone(), json!(doc.metadata));
        } else if !doc.metadata.is_empty() {
            return Err(format!(
                "Milvus collection {} has no JSON field {:?} for metadata",
                self.collection, self.metadata_field
            )
            .into());
        }
        if schema.has_namespace {
            // The partition key can't be null; "" means no namespace.
            row.insert(
                self.namespace_field.clone(),
                json!(namespace.unwrap_or_default()),
            );
        } else if namespace.is_some() {
            return Err(format!(
                "Milvus collection {} has no {:?} field for namespaces",
                self.collection, self.namespace_field
            )
            .into());
        }
        Ok(Value::Object(row))
    }

    pub(super) fn to_document(&self, mut hit: Map<String, Value>, schema: &Schema) -> Document {
        let distance = hit
            .remove("distance")
            .and_then(|d| d.as_f64())
            .unwrap_or_default();
        hit.remove(&schema.primary);
        let page_content = match hit.remove(&self.text_field) {
            Some(Value::String(text)) => text,
            Some(Value::Null) | None => String::new(),
            Some(other) => other.to_string(),
        };
        let mut metadata: HashMap<String, Value> = HashMap::new();
        match hit.remove(&self.metadata_field) {
            // REST returns JSON fields as strings.
            Some(Value::String(raw)) => match serde_json::from_str::<Value>(&raw) {
                Ok(Value::Object(map)) => metadata.extend(map),
                Ok(other) => {
                    metadata.insert(self.metadata_field.clone(), other);
                }
                Err(_) => {
                    metadata.insert(self.metadata_field.clone(), Value::String(raw));
                }
            },
            Some(Value::Object(map)) => metadata.extend(map),
            Some(Value::Null) | None => {}
            Some(other) => {
                metadata.insert(self.metadata_field.clone(), other);
            }
        }
        metadata.extend(hit.into_iter().filter(|(_, v)| !v.is_null()));
        Document {
            page_content,
            metadata,
            score: score(&schema.metric, distance),
        }
    }
}

#[async_trait]
impl VectorStore for Store {
    type Options = MilvusOptions;

    /// Embeds the documents and inserts them in batches, returning their ids.
    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &MilvusOptions,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        if docs.is_empty() {
            return Ok(vec![]);
        }
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
        let schema = self
            .schema(Some(vectors[0].len()))
            .await?
            .ok_or("Milvus collection is missing")?;
        let namespace = self.namespace(opt);

        let mut ids = Vec::with_capacity(docs.len());
        let pairs: Vec<_> = docs.iter().zip(&vectors).collect();
        for batch in pairs.chunks(BATCH_SIZE) {
            let own_ids: Vec<Option<String>> = batch
                .iter()
                .map(|_| (!schema.auto_id).then(|| uuid::Uuid::new_v4().to_string()))
                .collect();
            let rows = batch
                .iter()
                .zip(&own_ids)
                .map(|((doc, vector), id)| self.row(doc, vector, id.as_deref(), namespace, schema))
                .collect::<Result<Vec<_>, _>>()?;
            let answer = self
                .post("entities/insert", json!({ "data": rows }))
                .await?;
            if schema.auto_id {
                let inserted: Vec<String> = answer["data"]["insertIds"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .map(|id| match id {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .collect();
                if inserted.len() != batch.len() {
                    return Err(format!(
                        "Milvus returned {} ids for {} rows",
                        inserted.len(),
                        batch.len()
                    )
                    .into());
                }
                ids.extend(inserted);
            } else {
                ids.extend(own_ids.into_iter().flatten());
            }
        }
        Ok(ids)
    }

    /// Returns up to `limit` documents nearest to `query`, best first.
    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &MilvusOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        if limit == 0 {
            return Ok(vec![]);
        }
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let vector = embedder.embed_query(query).await?;
        let Some(schema) = self.schema(None).await? else {
            return Ok(vec![]); // Nothing was ever added.
        };
        let (filter, params) = self.filter(opt, schema)?;

        let mut body = json!({
            "data": [vector],
            "annsField": self.vector_field,
            "limit": limit,
            "outputFields": schema.output_fields,
        });
        if let Some(filter) = filter {
            body["filter"] = json!(filter);
            if !params.is_empty() {
                body["exprParams"] = Value::Object(params);
            }
        }
        let answer = self.post("entities/search", body).await?;
        Ok(answer["data"]
            .as_array()
            .map(Vec::as_slice)
            .unwrap_or_default()
            .iter()
            .filter_map(|hit| match hit {
                Value::Object(hit) => Some(self.to_document(hit.clone(), schema)),
                _ => None,
            })
            .filter(|doc| {
                opt.score_threshold
                    .is_none_or(|threshold| doc.score >= f64::from(threshold))
            })
            .collect())
    }
}
