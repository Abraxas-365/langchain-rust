use std::{
    collections::HashMap,
    error::Error,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use ::redis::{aio::ConnectionLike, aio::ConnectionManager, RedisError, Value as RedisValue};
use async_trait::async_trait;
use serde_json::Value;

use crate::{
    embedding::Embedder,
    schemas::Document,
    vectorstore::{VecStoreOptions, VectorStore},
};

/// Alias of the KNN distance in search replies. Not a field anyone stores.
const SCORE_FIELD: &str = "__vector_score";
/// Documents per HSET pipeline.
const WRITE_BATCH: usize = 500;

/// How the vector index compares embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RedisDistance {
    #[default]
    Cosine,
    L2,
    /// Inner product.
    IP,
}

impl RedisDistance {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cosine => "COSINE",
            Self::L2 => "L2",
            Self::IP => "IP",
        }
    }

    /// Redis returns a distance (lower is closer); `Document::score` is higher-is-better.
    /// COSINE and IP distances are `1 - similarity`, so this gives the similarity back.
    fn score(self, distance: f64) -> f64 {
        match self {
            Self::Cosine | Self::IP => 1.0 - distance,
            Self::L2 => -distance,
        }
    }
}

/// Vector index algorithm. HNSW is approximate and fast; FLAT is exact brute force.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorAlgorithm {
    #[default]
    Hnsw,
    Flat,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    /// Exact-match values (`@field:{value}`).
    Tag,
    /// Full-text (`@field:word`).
    Text,
    /// Ranges (`@field:[min max]`); returned as JSON numbers in metadata.
    Numeric,
}

/// A metadata field added to the index so it can be filtered on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetadataField {
    pub name: String,
    pub field_type: FieldType,
}

impl MetadataField {
    pub fn tag(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            field_type: FieldType::Tag,
        }
    }

    pub fn text(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            field_type: FieldType::Text,
        }
    }

    pub fn numeric(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            field_type: FieldType::Numeric,
        }
    }
}

/// A Redis vector store. See the [module docs](super).
///
/// Any async Redis connection works (`ConnectionManager`, `MultiplexedConnection`...).
pub struct Store<C = ConnectionManager> {
    pub(super) connection: C,
    pub(super) embedder: Arc<dyn Embedder>,
    pub(super) index_name: String,
    pub(super) key_prefix: String,
    pub(super) content_key: String,
    pub(super) vector_key: String,
    pub(super) namespace_key: String,
    pub(super) namespace: Option<String>,
    pub(super) metadata_schema: Vec<MetadataField>,
    pub(super) distance: RedisDistance,
    pub(super) algorithm: VectorAlgorithm,
    pub(super) index_ready: AtomicBool,
}

/// Escapes a value for a TAG query (`@field:{value}`): everything but letters, digits
/// and `_` gets a backslash, so `tenant-a` doesn't parse as `tenant` minus `a`.
pub(super) fn escape_tag(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for c in value.chars() {
        if !(c.is_alphanumeric() || c == '_') {
            escaped.push('\\');
        }
        escaped.push(c);
    }
    escaped
}

/// Field names go into the query unescaped, so only allow plain identifiers.
pub(super) fn is_field_name(name: &str) -> bool {
    let mut chars = name.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// A metadata value as stored in the hash, or `None` to leave it out.
fn hash_value(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Number(n) => Some(n.to_string()),
        other => Some(other.to_string()),
    }
}

fn is_missing_index(err: &RedisError) -> bool {
    let text = err.to_string().to_lowercase();
    err.code() == Some("SEARCH_INDEX_NOT_FOUND")
        || text.contains("unknown index name")
        || text.contains("no such index")
        || text.contains("index not found")
}

fn is_existing_index(err: &RedisError) -> bool {
    err.code() == Some("SEARCH_INDEX_EXISTS") || err.to_string().contains("Index already exists")
}

fn redis_string(value: &RedisValue) -> Option<String> {
    match value {
        RedisValue::BulkString(bytes) => String::from_utf8(bytes.clone()).ok(),
        RedisValue::SimpleString(s) => Some(s.clone()),
        RedisValue::VerbatimString { text, .. } => Some(text.clone()),
        RedisValue::Int(i) => Some(i.to_string()),
        RedisValue::Double(d) => Some(d.to_string()),
        _ => None,
    }
}

/// `[k, v, k, v, ...]` (RESP2) or a map (RESP3) as pairs.
fn pairs(value: &RedisValue) -> Option<Vec<(&RedisValue, &RedisValue)>> {
    match value {
        RedisValue::Array(items) if items.len() % 2 == 0 => {
            Some(items.chunks(2).map(|kv| (&kv[0], &kv[1])).collect())
        }
        RedisValue::Map(entries) => Some(entries.iter().map(|(k, v)| (k, v)).collect()),
        _ => None,
    }
}

/// Field/value pairs with string names.
fn fields(value: &RedisValue) -> Option<Vec<(String, &RedisValue)>> {
    pairs(value)?
        .into_iter()
        .map(|(k, v)| Some((redis_string(k)?, v)))
        .collect()
}

/// A search result: the document key and its returned fields.
pub(super) type Hit<'a> = (String, Vec<(String, &'a RedisValue)>);

/// The documents of an FT.SEARCH reply as `(key, fields)`, in either protocol:
/// RESP2 `[total, key, [field, value, ...], ...]`, RESP3
/// `{results: [{id, extra_attributes: {field: value}}]}`.
pub(super) fn search_hits(reply: &RedisValue) -> Result<Vec<Hit<'_>>, Box<dyn Error>> {
    let malformed =
        || -> Box<dyn Error> { format!("unexpected FT.SEARCH reply: {reply:?}").into() };
    match reply {
        RedisValue::Array(items) => items[1.min(items.len())..]
            .chunks(2)
            .map(|hit| match hit {
                [key, values] => Some((redis_string(key)?, fields(values)?)),
                _ => None,
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(malformed),
        RedisValue::Map(_) => {
            let top = pairs(reply).ok_or_else(malformed)?;
            let results = top
                .iter()
                .find(|(k, _)| redis_string(k).as_deref() == Some("results"))
                .map(|(_, v)| *v)
                .ok_or_else(malformed)?;
            let RedisValue::Array(results) = results else {
                return Err(malformed());
            };
            results
                .iter()
                .map(|hit| {
                    let hit = pairs(hit)?;
                    let get = |name: &str| {
                        hit.iter()
                            .find(|(k, _)| redis_string(k).as_deref() == Some(name))
                            .map(|(_, v)| *v)
                    };
                    Some((redis_string(get("id")?)?, fields(get("extra_attributes")?)?))
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(malformed)
        }
        _ => Err(malformed()),
    }
}

impl<C> Store<C>
where
    C: ConnectionLike + Clone + Send + Sync,
{
    /// The RediSearch index name.
    pub fn index_name(&self) -> &str {
        &self.index_name
    }

    /// Prefix of the document keys.
    pub fn key_prefix(&self) -> &str {
        &self.key_prefix
    }

    pub(super) fn create_index_cmd(&self, dim: usize) -> ::redis::Cmd {
        let mut cmd = ::redis::cmd("FT.CREATE");
        cmd.arg(&self.index_name)
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg(1)
            .arg(&self.key_prefix)
            .arg("SCHEMA")
            .arg(&self.content_key)
            .arg("TEXT")
            .arg(&self.namespace_key)
            .arg("TAG");
        for field in &self.metadata_schema {
            cmd.arg(&field.name).arg(match field.field_type {
                FieldType::Tag => "TAG",
                FieldType::Text => "TEXT",
                FieldType::Numeric => "NUMERIC",
            });
        }
        cmd.arg(&self.vector_key)
            .arg("VECTOR")
            .arg(match self.algorithm {
                VectorAlgorithm::Hnsw => "HNSW",
                VectorAlgorithm::Flat => "FLAT",
            })
            .arg(6)
            .arg("TYPE")
            .arg("FLOAT32")
            .arg("DIM")
            .arg(dim)
            .arg("DISTANCE_METRIC")
            .arg(self.distance.as_str());
        cmd
    }

    /// Creates the index (with the dimension of the first embeddings) unless it exists.
    async fn ensure_index(&self, dim: usize) -> Result<(), Box<dyn Error>> {
        if self.index_ready.load(Ordering::Acquire) {
            return Ok(());
        }
        let mut connection = self.connection.clone();
        let info: Result<RedisValue, RedisError> = ::redis::cmd("FT.INFO")
            .arg(&self.index_name)
            .query_async(&mut connection)
            .await;
        match info {
            Ok(_) => {}
            Err(err) if is_missing_index(&err) => {
                let created: Result<(), RedisError> = self
                    .create_index_cmd(dim)
                    .query_async(&mut connection)
                    .await;
                match created {
                    Ok(()) => {}
                    // Another writer created it in between.
                    Err(err) if is_existing_index(&err) => {}
                    Err(err) => return Err(err.into()),
                }
            }
            Err(err) => return Err(err.into()),
        }
        self.index_ready.store(true, Ordering::Release);
        Ok(())
    }

    /// The pre-filter of the KNN query: namespace TAG plus the user filter.
    ///
    /// A string filter is a raw query expression (`@year:[2020 +inf]`); an object is an
    /// AND of equalities, numbers as `@k:[n n]` and everything else as `@k:{value}` TAGs.
    pub(super) fn filter_expression(
        &self,
        namespace: Option<&str>,
        filters: Option<&Value>,
    ) -> Result<String, Box<dyn Error>> {
        let mut parts = Vec::new();
        if let Some(ns) = namespace {
            parts.push(format!("@{}:{{{}}}", self.namespace_key, escape_tag(ns)));
        }
        match filters {
            None | Some(Value::Null) => {}
            Some(Value::String(raw)) if raw.trim().is_empty() || raw.trim() == "*" => {}
            Some(Value::String(raw)) => parts.push(format!("({raw})")),
            Some(Value::Object(fields)) => {
                let mut fields: Vec<_> = fields.iter().collect();
                fields.sort_by_key(|(k, _)| *k);
                for (key, value) in fields {
                    if !is_field_name(key) {
                        return Err(format!("invalid Redis filter field {key:?}").into());
                    }
                    match value {
                        Value::Number(n) => parts.push(format!("@{key}:[{n} {n}]")),
                        Value::String(s) => parts.push(format!("@{key}:{{{}}}", escape_tag(s))),
                        Value::Bool(b) => parts.push(format!("@{key}:{{{b}}}")),
                        other => {
                            return Err(format!(
                                "unsupported Redis filter value for {key:?}: {other}"
                            )
                            .into())
                        }
                    }
                }
            }
            Some(other) => {
                return Err(format!(
                    "Redis filters are a query string or an object of equalities, got {other}"
                )
                .into())
            }
        }
        Ok(if parts.is_empty() {
            "*".to_string()
        } else {
            format!("({})", parts.join(" "))
        })
    }

    pub(super) fn search_cmd(&self, filter: &str, limit: usize, vector: &[f64]) -> ::redis::Cmd {
        let mut cmd = ::redis::cmd("FT.SEARCH");
        cmd.arg(&self.index_name)
            .arg(format!(
                "{filter}=>[KNN {limit} @{} $vector AS {SCORE_FIELD}]",
                self.vector_key
            ))
            .arg("PARAMS")
            .arg(2)
            .arg("vector")
            .arg(vector_bytes(vector))
            .arg("SORTBY")
            .arg(SCORE_FIELD)
            .arg("ASC")
            // FT.SEARCH returns 10 results unless told otherwise.
            .arg("LIMIT")
            .arg(0)
            .arg(limit)
            .arg("DIALECT")
            .arg(2);
        cmd
    }

    fn document(
        &self,
        key: &str,
        fields: Vec<(String, &RedisValue)>,
    ) -> Result<Document, Box<dyn Error>> {
        let mut page_content = String::new();
        let mut distance = None;
        let mut metadata = HashMap::new();
        for (name, value) in fields {
            if name == self.vector_key {
                continue;
            }
            let Some(text) = redis_string(value) else {
                continue;
            };
            if name == self.content_key {
                page_content = text;
            } else if name == SCORE_FIELD {
                distance = Some(text.parse::<f64>()?);
            } else {
                let numeric = self
                    .metadata_schema
                    .iter()
                    .any(|f| f.name == name && f.field_type == FieldType::Numeric);
                let value = match text.parse::<serde_json::Number>() {
                    Ok(n) if numeric => Value::Number(n),
                    _ => Value::String(text),
                };
                metadata.insert(name, value);
            }
        }
        let distance = distance.ok_or_else(|| format!("FT.SEARCH reply without {SCORE_FIELD}"))?;
        Ok(Document {
            page_content,
            metadata,
            score: self.distance.score(distance),
            id: Some(
                key.strip_prefix(&self.key_prefix)
                    .unwrap_or(key)
                    .to_string(),
            ),
        })
    }
}

pub(super) fn vector_bytes(vector: &[f64]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|x| (*x as f32).to_le_bytes())
        .collect()
}

#[async_trait]
impl<C> VectorStore for Store<C>
where
    C: ConnectionLike + Clone + Send + Sync,
{
    type Options = VecStoreOptions<Value>;

    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &Self::Options,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        if docs.is_empty() {
            return Ok(Vec::new());
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
        self.ensure_index(vectors[0].len()).await?;

        let namespace = opt.name_space.as_deref().or(self.namespace.as_deref());
        let keys: Vec<String> = docs
            .iter()
            .map(|_| format!("{}{}", self.key_prefix, uuid::Uuid::new_v4()))
            .collect();
        let mut connection = self.connection.clone();
        for start in (0..docs.len()).step_by(WRITE_BATCH) {
            let end = (start + WRITE_BATCH).min(docs.len());
            let mut pipe = ::redis::pipe();
            for i in start..end {
                let doc = &docs[i];
                let cmd = pipe.cmd("HSET").arg(&keys[i]);
                cmd.arg(&self.content_key)
                    .arg(&doc.page_content)
                    .arg(&self.vector_key)
                    .arg(vector_bytes(&vectors[i]));
                if let Some(ns) = namespace {
                    cmd.arg(&self.namespace_key).arg(ns);
                }
                let mut metadata: Vec<_> = doc.metadata.iter().collect();
                metadata.sort_by_key(|(k, _)| *k);
                for (key, value) in metadata {
                    // Reserved fields would overwrite the document itself.
                    if [&self.content_key, &self.vector_key, &self.namespace_key].contains(&key)
                        || key == SCORE_FIELD
                    {
                        continue;
                    }
                    if let Some(value) = hash_value(value) {
                        cmd.arg(key).arg(value);
                    }
                }
                cmd.ignore();
            }
            let () = pipe.query_async(&mut connection).await?;
        }
        Ok(keys)
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
        let filter = self.filter_expression(namespace, opt.filters.as_ref())?;
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let vector = embedder.embed_query(query).await?;

        let mut connection = self.connection.clone();
        let reply: Result<RedisValue, RedisError> = self
            .search_cmd(&filter, limit, &vector)
            .query_async(&mut connection)
            .await;
        let reply = match reply {
            Ok(reply) => reply,
            // Nothing was written yet.
            Err(err) if is_missing_index(&err) => return Ok(Vec::new()),
            Err(err) => return Err(err.into()),
        };
        let mut docs = Vec::new();
        for (key, fields) in search_hits(&reply)? {
            let doc = self.document(&key, fields)?;
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

#[cfg(test)]
mod unit_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tag_escaping_and_field_names() {
        assert_eq!(escape_tag("tenant-a"), r"tenant\-a");
        assert_eq!(escape_tag("a b.c{d}|e"), r"a\ b\.c\{d\}\|e");
        assert_eq!(escape_tag("año_1"), "año_1");
        assert!(is_field_name("source_2"));
        assert!(is_field_name("_x"));
        for bad in ["", "2x", "a-b", "a b", "a}"] {
            assert!(!is_field_name(bad), "{bad}");
        }
    }

    #[test]
    fn scores_are_higher_is_better() {
        assert_eq!(RedisDistance::Cosine.score(0.25), 0.75);
        assert_eq!(RedisDistance::IP.score(0.25), 0.75);
        assert_eq!(RedisDistance::L2.score(0.25), -0.25);
    }

    #[test]
    fn vectors_are_little_endian_f32() {
        assert_eq!(vector_bytes(&[1.0, -2.5]), [0, 0, 128, 63, 0, 0, 32, 192]);
    }

    #[test]
    fn hash_values() {
        assert_eq!(hash_value(&json!("a")), Some("a".into()));
        assert_eq!(hash_value(&json!(2.5)), Some("2.5".into()));
        assert_eq!(hash_value(&json!(true)), Some("true".into()));
        assert_eq!(hash_value(&json!(["x", 1])), Some(r#"["x",1]"#.into()));
        assert_eq!(hash_value(&json!(null)), None);
    }
}
