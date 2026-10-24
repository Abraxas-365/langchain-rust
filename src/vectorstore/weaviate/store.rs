use std::{
    error::Error,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use async_trait::async_trait;
use reqwest::{Client, Method, StatusCode};
use serde_json::{json, Map, Value};

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{VecStoreOptions, VectorStore},
};

/// Objects per `/v1/batch/objects` request.
pub(super) const BATCH_SIZE: usize = 100;

/// Weaviate property types that can be selected in GraphQL without sub-fields.
const SELECTABLE_TYPES: &[&str] = &[
    "text", "string", "int", "number", "boolean", "date", "uuid", "blob",
];

/// [Weaviate](https://weaviate.io/) vector store over the REST and GraphQL APIs.
/// Build it with [`StoreBuilder`](super::StoreBuilder).
///
/// - Vectors come from the embedder (bring-your-own-vectors), so the collection needs no
///   vectorizer module. The first write creates the collection if it's missing (no
///   vectorizer, `text_key` and `namespace_key` text properties) or adds `namespace_key` to
///   an existing one. The namespace property uses `field` tokenization so namespace filters
///   are exact matches. Other properties come from Weaviate's auto-schema.
/// - The document text is stored in the `text_key` property (default `text`), and each
///   metadata entry becomes a property. Metadata keys must be valid GraphQL names
///   (`[_A-Za-z][_0-9A-Za-z]*`). Nested objects and mixed arrays are stored as JSON strings,
///   and nulls are dropped.
/// - Options:
///   - `filters`: a Weaviate `where` filter in its JSON form, e.g.
///     `json!({"path": ["source"], "operator": "Equal", "valueText": "a.txt"})`.
///   - `name_space`: stored in the `namespace_key` property on writes and required to match
///     on searches. It falls back to the builder's `namespace`.
///   - `score_threshold`: keeps `score >= threshold`.
///   - `embedder`: per-call override.
/// - `Document::score` is the certainty (`1 - distance / 2`) for cosine collections and
///   `-distance` for the other metrics, so higher is always better.
pub struct Store {
    pub(super) client: Client,
    pub(super) url: String,
    pub(super) api_key: Option<String>,
    pub(super) class_name: String,
    pub(super) embedder: Arc<dyn Embedder>,
    pub(super) text_key: String,
    pub(super) namespace_key: String,
    pub(super) namespace: Option<String>,
    pub(super) metadata_fields: Option<Vec<String>>,
    pub(super) schema_ready: AtomicBool,
}

pub type WeaviateOptions = VecStoreOptions<Value>;

pub(super) fn is_graphql_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c == '_' || c.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

/// Renders a JSON value as a GraphQL input literal. Object keys are unquoted (and sorted,
/// so the output doesn't depend on serde_json's `preserve_order`), and the value of
/// `operator` is an enum, so it's emitted bare.
pub(super) fn graphql_literal(value: &Value) -> Result<String, Box<dyn Error>> {
    Ok(match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => value.to_string(),
        Value::Array(items) => format!(
            "[{}]",
            items
                .iter()
                .map(graphql_literal)
                .collect::<Result<Vec<_>, _>>()?
                .join(", ")
        ),
        Value::Object(fields) => {
            let mut fields: Vec<_> = fields.iter().collect();
            fields.sort_by_key(|(key, _)| *key);
            let mut parts = Vec::with_capacity(fields.len());
            for (key, value) in fields {
                if !is_graphql_name(key) {
                    return Err(format!("invalid Weaviate filter key {key:?}").into());
                }
                let rendered = match (key.as_str(), value) {
                    ("operator", Value::String(op)) if is_graphql_name(op) => op.clone(),
                    ("operator", _) => {
                        return Err(format!("invalid Weaviate filter operator {value}").into())
                    }
                    _ => graphql_literal(value)?,
                };
                parts.push(format!("{key}: {rendered}"));
            }
            format!("{{{}}}", parts.join(", "))
        }
    })
}

/// Weaviate properties can be scalars or arrays of one scalar type.
fn weaviate_property_value(value: &Value) -> Option<Value> {
    fn kind(v: &Value) -> Option<u8> {
        match v {
            Value::Bool(_) => Some(0),
            Value::Number(_) => Some(1),
            Value::String(_) => Some(2),
            _ => None,
        }
    }
    match value {
        Value::Null => None,
        Value::Bool(_) | Value::Number(_) | Value::String(_) => Some(value.clone()),
        Value::Array(items)
            if !items.is_empty()
                && kind(&items[0]).is_some()
                && items.iter().all(|i| kind(i) == kind(&items[0])) =>
        {
            Some(value.clone())
        }
        Value::Array(_) | Value::Object(_) => Some(Value::String(value.to_string())),
    }
}

fn weaviate_error(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    let messages: Vec<&str> = value["error"]
        .as_array()
        .or_else(|| value["errors"].as_array())?
        .iter()
        .filter_map(|e| e["message"].as_str())
        .collect();
    (!messages.is_empty()).then(|| messages.join("; "))
}

impl Store {
    async fn send(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<(StatusCode, String), Box<dyn Error>> {
        let mut request = self.client.request(method, format!("{}{path}", self.url));
        if let Some(key) = &self.api_key {
            request = request.bearer_auth(key);
        }
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request.send().await?;
        let status = response.status();
        Ok((status, response.text().await?))
    }

    async fn request(
        &self,
        method: Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<Value, Box<dyn Error>> {
        let (status, text) = self.send(method, path, body).await?;
        if !status.is_success() {
            let message = weaviate_error(&text).unwrap_or(text);
            return Err(format!("Weaviate error ({status}): {message}").into());
        }
        Ok(serde_json::from_str(&text).unwrap_or(Value::Null))
    }

    /// The collection schema, or `None` if the collection doesn't exist yet.
    async fn schema(&self) -> Result<Option<Value>, Box<dyn Error>> {
        let path = format!("/v1/schema/{}", self.class_name);
        let (status, text) = self.send(Method::GET, &path, None).await?;
        if status == StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            let message = weaviate_error(&text).unwrap_or(text);
            return Err(format!("Weaviate error ({status}): {message}").into());
        }
        Ok(Some(serde_json::from_str(&text)?))
    }

    /// POSTs a schema change; "already exists" (another writer got there first) is fine.
    async fn create_schema(&self, path: &str, body: &Value) -> Result<(), Box<dyn Error>> {
        let (status, text) = self.send(Method::POST, path, Some(body)).await?;
        if status.is_success() || (status.is_client_error() && text.contains("already exists")) {
            return Ok(());
        }
        let message = weaviate_error(&text).unwrap_or(text);
        Err(format!("Weaviate error ({status}): {message}").into())
    }

    /// Creates the collection, or adds the namespace property, before the first write.
    ///
    /// Weaviate's auto-schema would make the namespace a `word`-tokenized text property,
    /// and then an `Equal` filter on `tenant` also matches `tenant-b`. `field` tokenization
    /// makes it an exact match. Collections created here also turn off the stopword preset,
    /// because Weaviate rejects filters whose value is a stopword (a namespace named `a`).
    async fn ensure_schema(&self) -> Result<(), Box<dyn Error>> {
        if self.schema_ready.load(Ordering::Acquire) {
            return Ok(());
        }
        let namespace_property = json!({
            "name": self.namespace_key,
            "dataType": ["text"],
            "tokenization": "field",
        });
        let schema = self.schema().await?;
        match schema {
            None => {
                let class = json!({
                    "class": self.class_name,
                    "vectorizer": "none",
                    "invertedIndexConfig": { "stopwords": { "preset": "none" } },
                    "properties": [
                        { "name": self.text_key, "dataType": ["text"] },
                        namespace_property,
                    ],
                });
                self.create_schema("/v1/schema", &class).await?;
            }
            Some(schema) => {
                let has_namespace = schema["properties"]
                    .as_array()
                    .is_some_and(|ps| ps.iter().any(|p| p["name"] == json!(self.namespace_key)));
                if !has_namespace {
                    let path = format!("/v1/schema/{}/properties", self.class_name);
                    self.create_schema(&path, &namespace_property).await?;
                }
            }
        }
        self.schema_ready.store(true, Ordering::Release);
        Ok(())
    }

    fn namespace<'a>(&'a self, opt: &'a WeaviateOptions) -> Option<&'a str> {
        opt.name_space.as_deref().or(self.namespace.as_deref())
    }

    fn where_filter(&self, opt: &WeaviateOptions) -> Option<Value> {
        let namespace = self.namespace(opt).map(
            |ns| json!({ "path": [self.namespace_key], "operator": "Equal", "valueText": ns }),
        );
        match (namespace, opt.filters.clone()) {
            (Some(ns), Some(filter)) => {
                Some(json!({ "operator": "And", "operands": [ns, filter] }))
            }
            (ns, filter) => ns.or(filter),
        }
    }

    fn properties(&self, doc: &Document, namespace: Option<&str>) -> Map<String, Value> {
        let mut properties: Map<String, Value> = doc
            .metadata
            .iter()
            .filter_map(|(k, v)| weaviate_property_value(v).map(|v| (k.clone(), v)))
            .collect();
        properties.insert(self.text_key.clone(), json!(doc.page_content));
        if let Some(ns) = namespace {
            properties.insert(self.namespace_key.clone(), json!(ns));
        }
        properties
    }

    /// Properties to select: the configured `metadata_fields`, or every selectable
    /// property in the collection's schema. `None` if the collection doesn't exist.
    async fn fields(&self) -> Result<Option<Vec<String>>, Box<dyn Error>> {
        let mut fields = match &self.metadata_fields {
            Some(fields) => fields.clone(),
            None => {
                let Some(schema) = self.schema().await? else {
                    return Ok(None);
                };
                schema["properties"]
                    .as_array()
                    .map(Vec::as_slice)
                    .unwrap_or_default()
                    .iter()
                    .filter(|p| {
                        p["dataType"][0]
                            .as_str()
                            .is_some_and(|t| SELECTABLE_TYPES.contains(&t.trim_end_matches("[]")))
                    })
                    .filter_map(|p| p["name"].as_str().map(str::to_string))
                    .collect()
            }
        };
        if !fields.contains(&self.text_key) {
            fields.insert(0, self.text_key.clone());
        }
        if let Some(bad) = fields.iter().find(|f| !is_graphql_name(f)) {
            return Err(format!("invalid Weaviate property name {bad:?}").into());
        }
        Ok(Some(fields))
    }
}

#[async_trait]
impl VectorStore for Store {
    type Options = WeaviateOptions;

    /// Embeds the documents and inserts them in batches, returning their new ids.
    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &WeaviateOptions,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        if docs.is_empty() {
            return Ok(vec![]);
        }
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let texts: Vec<String> = docs.iter().map(|d| d.page_content.clone()).collect();
        let embeddings = embedder.embed_documents(&texts).await?;
        if embeddings.len() != docs.len() {
            return Err(format!(
                "embedder returned {} vectors for {} documents",
                embeddings.len(),
                docs.len()
            )
            .into());
        }
        self.ensure_schema().await?;

        let namespace = self.namespace(opt);
        let ids: Vec<String> = docs
            .iter()
            .map(|_| uuid::Uuid::new_v4().to_string())
            .collect();
        let objects: Vec<Value> = docs
            .iter()
            .zip(&embeddings)
            .zip(&ids)
            .map(|((doc, vector), id)| {
                json!({
                    "class": self.class_name,
                    "id": id,
                    "vector": vector,
                    "properties": self.properties(doc, namespace),
                })
            })
            .collect();

        for batch in objects.chunks(BATCH_SIZE) {
            let response = self
                .request(
                    Method::POST,
                    "/v1/batch/objects",
                    Some(&json!({ "objects": batch })),
                )
                .await?;
            // The batch endpoint answers 200 and reports failures per object.
            let failures: Vec<String> = response
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or_default()
                .iter()
                .filter_map(|object| weaviate_error(&object["result"]["errors"].to_string()))
                .collect();
            if !failures.is_empty() {
                return Err(format!(
                    "Weaviate rejected {} object(s): {}",
                    failures.len(),
                    failures.join("; ")
                )
                .into());
            }
        }
        Ok(ids)
    }

    /// Returns up to `limit` documents nearest to `query`, best first.
    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &WeaviateOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let vector = embedder.embed_query(query).await?;
        let Some(fields) = self.fields().await? else {
            return Ok(vec![]); // Nothing was ever added.
        };

        let mut arguments = format!(
            "nearVector: {{vector: {}}}, limit: {limit}",
            graphql_literal(&json!(vector))?
        );
        if let Some(filter) = self.where_filter(opt) {
            arguments.push_str(&format!(", where: {}", graphql_literal(&filter)?));
        }
        let graphql = format!(
            "{{ Get {{ {class}({arguments}) {{ {fields} _additional {{ id certainty distance }} }} }} }}",
            class = self.class_name,
            fields = fields.join(" "),
        );

        let response = self
            .request(
                Method::POST,
                "/v1/graphql",
                Some(&json!({ "query": graphql })),
            )
            .await?;
        if let Some(message) = weaviate_error(&response.to_string()) {
            return Err(format!("Weaviate GraphQL error: {message}").into());
        }

        let hits = response["data"]["Get"][&self.class_name]
            .as_array()
            .cloned()
            .unwrap_or_default();
        Ok(hits
            .into_iter()
            .filter_map(|hit| {
                let Value::Object(mut properties) = hit else {
                    return None;
                };
                let additional = properties.remove("_additional").unwrap_or_default();
                let score = additional["certainty"]
                    .as_f64()
                    .or_else(|| additional["distance"].as_f64().map(|d| -d))
                    .unwrap_or_default();
                let id = additional["id"].as_str().map(str::to_string);
                let page_content = match properties.remove(&self.text_key) {
                    Some(Value::String(text)) => text,
                    Some(Value::Null) | None => String::new(),
                    Some(other) => other.to_string(),
                };
                Some(Document {
                    page_content,
                    metadata: properties
                        .into_iter()
                        .filter(|(_, v)| !v.is_null())
                        .collect(),
                    score,
                    id,
                })
            })
            .filter(|doc| {
                opt.score_threshold
                    .is_none_or(|threshold| doc.score >= f64::from(threshold))
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graphql_literals() {
        let filter = json!({
            "operator": "And",
            "operands": [
                { "path": ["source"], "operator": "Equal", "valueText": "a \"b\"\n" },
                { "path": ["n"], "operator": "GreaterThan", "valueNumber": 1.5 },
            ],
        });
        assert_eq!(
            graphql_literal(&filter).unwrap(),
            r#"{operands: [{operator: Equal, path: ["source"], valueText: "a \"b\"\n"}, {operator: GreaterThan, path: ["n"], valueNumber: 1.5}], operator: And}"#
        );
        assert_eq!(
            graphql_literal(&json!([0.5, -1, true, null])).unwrap(),
            "[0.5, -1, true, null]"
        );

        for bad in [
            json!({ "bad-key": 1 }),
            json!({ "operator": "Equal) { x }" }),
            json!({ "operator": 1 }),
        ] {
            assert!(graphql_literal(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn graphql_names() {
        for ok in ["text", "_x", "nameSpace", "a1"] {
            assert!(is_graphql_name(ok), "{ok}");
        }
        for bad in ["", "1a", "a-b", "a b", "a.b", "é"] {
            assert!(!is_graphql_name(bad), "{bad}");
        }
    }

    #[test]
    fn property_values_weaviate_cannot_store_are_converted_or_dropped() {
        for (input, expected) in [
            (json!("s"), Some(json!("s"))),
            (json!(2), Some(json!(2))),
            (json!(true), Some(json!(true))),
            (json!(["a", "b"]), Some(json!(["a", "b"]))),
            (json!([1, 2.5]), Some(json!([1, 2.5]))),
            (json!([]), Some(json!("[]"))),
            (json!([1, "a"]), Some(json!("[1,\"a\"]"))),
            (json!([{ "a": 1 }]), Some(json!("[{\"a\":1}]"))),
            (json!({ "a": 1 }), Some(json!("{\"a\":1}"))),
            (json!(null), None),
        ] {
            assert_eq!(weaviate_property_value(&input), expected, "{input}");
        }
    }

    #[test]
    fn error_messages() {
        assert_eq!(
            weaviate_error(r#"{"error":[{"message":"a"},{"message":"b"}]}"#).as_deref(),
            Some("a; b")
        );
        assert_eq!(
            weaviate_error(r#"{"errors":[{"message":"gql"}]}"#).as_deref(),
            Some("gql")
        );
        assert_eq!(weaviate_error(r#"{"errors":[]}"#), None);
        assert_eq!(weaviate_error("null"), None);
        assert_eq!(weaviate_error("not json"), None);
    }
}
