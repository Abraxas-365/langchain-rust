use std::{collections::HashMap, error::Error, sync::Arc};

use async_trait::async_trait;
use serde_json::Value;
use sqlx::{mysql::MySqlRow, AssertSqlSafe, MySql, Pool, Row};
use uuid::Uuid;

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{
        vectorstore::{document_ids, in_id_order},
        VecStoreOptions, VectorStore,
    },
};

/// Options for the MariaDB store. `filters` is a JSON object matched against document
/// metadata with `JSON_CONTAINS`: every key must be present with an equal value (nested
/// objects match as subsets, and a scalar matches an element of an array).
pub type MariaDBOptions = VecStoreOptions<Value>;

/// Longest document id the unique `varchar` column holds. (Vector indexes need a short
/// primary key, so rows get a separate `BIGINT` one.)
pub(crate) const MAX_ID_CHARS: usize = 255;
/// Keeps `IN (...)` lists well under the server's placeholder limit.
const ID_CHUNK: usize = 1000;

/// Vector store on MariaDB 11.7+ (native `VECTOR` column with an HNSW cosine index),
/// langchaingo's `mariadb` store. Build it with [`StoreBuilder`](super::StoreBuilder).
///
/// - Documents keep their [`Document::id`] when set (re-adding an id updates the row);
///   otherwise a UUID is generated.
/// - `similarity_search` returns `score` = cosine similarity (`1 - cosine distance`,
///   higher is closer). `score_threshold` (0 to 1) drops results below it, `name_space`
///   searches another collection in the same tables, and `filters` matches metadata.
/// - Writes and id lookups only accept the default options (plus `embedder` for writes).
pub struct Store {
    pub pool: Pool<MySql>,
    pub(crate) embedder: Arc<dyn Embedder>,
    pub(crate) vector_dimensions: usize,
    pub(crate) collection_name: String,
    pub(crate) collection_id: String,
    pub(crate) collection_table_name: String,
    pub(crate) embedding_table_name: String,
}

impl Store {
    pub(crate) async fn create_tables(&self) -> Result<(), Box<dyn Error>> {
        let coll = &self.collection_table_name;
        let emb = &self.embedding_table_name;
        let dims = self.vector_dimensions;
        sqlx::query(AssertSqlSafe(format!(
            "CREATE TABLE IF NOT EXISTS {coll} (
                id VARCHAR(36) NOT NULL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                cmetadata JSON
            )"
        )))
        .execute(&self.pool)
        .await?;
        sqlx::query(AssertSqlSafe(format!(
            "CREATE TABLE IF NOT EXISTS {emb} (
                pk BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                id VARCHAR({MAX_ID_CHARS}) NOT NULL UNIQUE,
                collection_id VARCHAR(36) NOT NULL,
                embedding VECTOR({dims}) NOT NULL,
                document LONGTEXT NOT NULL,
                cmetadata JSON,
                INDEX (collection_id),
                VECTOR INDEX (embedding) DISTANCE=cosine,
                FOREIGN KEY (collection_id) REFERENCES {coll} (id) ON DELETE CASCADE
            )"
        )))
        .execute(&self.pool)
        .await?;

        // An existing table keeps its size; catch a mismatch now rather than on insert.
        let column_type: String = sqlx::query(
            "SELECT CAST(COLUMN_TYPE AS CHAR) FROM information_schema.COLUMNS
             WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = ? AND COLUMN_NAME = 'embedding'",
        )
        .bind(emb)
        .fetch_one(&self.pool)
        .await?
        .try_get(0)?;
        if !column_type.eq_ignore_ascii_case(&format!("vector({dims})")) {
            return Err(format!(
                "table {emb} stores {column_type} but vector_dimensions is {dims}"
            )
            .into());
        }
        Ok(())
    }

    /// Drops both tables, with every collection in them.
    pub async fn drop_tables(&self) -> Result<(), Box<dyn Error>> {
        for table in [&self.embedding_table_name, &self.collection_table_name] {
            sqlx::query(AssertSqlSafe(format!("DROP TABLE IF EXISTS {table}")))
                .execute(&self.pool)
                .await?;
        }
        Ok(())
    }

    fn check_vector(&self, vector: &[f64]) -> Result<(), Box<dyn Error>> {
        if vector.len() != self.vector_dimensions {
            return Err(format!(
                "embedder returned {} dimensions, the store expects {}",
                vector.len(),
                self.vector_dimensions
            )
            .into());
        }
        Ok(())
    }

    fn row_to_document(row: &MySqlRow, score: f64) -> Result<Document, Box<dyn Error>> {
        let id: String = row.try_get("id")?;
        let page_content: String = row.try_get("document")?;
        let metadata: Option<String> = row.try_get("cmetadata")?;
        let metadata: HashMap<String, Value> = match metadata.as_deref() {
            None | Some("") => HashMap::new(),
            Some(text) => match serde_json::from_str(text)? {
                Value::Object(map) => map.into_iter().collect(),
                _ => HashMap::new(),
            },
        };
        Ok(Document {
            page_content,
            metadata,
            score,
            id: Some(id),
        })
    }
}

/// Table names go into SQL text, so only plain identifiers are allowed.
pub(crate) fn check_identifier(what: &str, name: &str) -> Result<(), Box<dyn Error>> {
    let valid = !name.is_empty()
        && name.len() <= 64
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        && !name.as_bytes()[0].is_ascii_digit();
    if !valid {
        return Err(format!(
            "{what} {name:?} must be 1 to 64 letters, digits or '_', not starting with a digit"
        )
        .into());
    }
    Ok(())
}

/// `[1,0.5,-2]`, the text `VEC_FromText` parses. Values are stored as f32.
pub(crate) fn vector_text(vector: &[f64]) -> String {
    let items: Vec<String> = vector.iter().map(|x| (*x as f32).to_string()).collect();
    format!("[{}]", items.join(","))
}

/// The metadata filter as JSON text for `JSON_CONTAINS`, or `None` for no filter.
pub(crate) fn filter_json(opt: &MariaDBOptions) -> Result<Option<String>, Box<dyn Error>> {
    match &opt.filters {
        None => Ok(None),
        Some(Value::Object(map)) if map.is_empty() => Ok(None),
        Some(filter @ Value::Object(_)) => Ok(Some(filter.to_string())),
        Some(other) => Err(format!("filters must be a JSON object, got {other}").into()),
    }
}

pub(crate) fn score_threshold(opt: &MariaDBOptions) -> Result<Option<f64>, Box<dyn Error>> {
    match opt.score_threshold {
        None => Ok(None),
        Some(t) if (0.0..=1.0).contains(&t) => Ok(Some(f64::from(t))),
        Some(t) => Err(format!("score_threshold must be between 0 and 1, got {t}").into()),
    }
}

fn reject_query_options(opt: &MariaDBOptions, method: &str) -> Result<(), Box<dyn Error>> {
    if opt.filters.is_some() || opt.score_threshold.is_some() || opt.name_space.is_some() {
        return Err(format!(
            "filters, score_threshold and name_space are not supported in mariadb {method}"
        )
        .into());
    }
    Ok(())
}

fn placeholders(n: usize) -> String {
    vec!["?"; n].join(",")
}

#[async_trait]
impl VectorStore for Store {
    type Options = MariaDBOptions;

    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &Self::Options,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        reject_query_options(opt, "add_documents")?;
        let ids = document_ids(docs, || Uuid::new_v4().to_string())?;
        if let Some(id) = ids.iter().find(|id| id.chars().count() > MAX_ID_CHARS) {
            return Err(
                format!("document id is longer than {MAX_ID_CHARS} characters: {id}").into(),
            );
        }
        if docs.is_empty() {
            return Ok(ids);
        }

        let texts: Vec<String> = docs.iter().map(|d| d.page_content.clone()).collect();
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let vectors = embedder.embed_documents(&texts).await?;
        if vectors.len() != docs.len() {
            return Err(format!(
                "embedder returned {} vectors for {} documents",
                vectors.len(),
                docs.len()
            )
            .into());
        }
        for vector in &vectors {
            self.check_vector(vector)?;
        }

        let emb = &self.embedding_table_name;
        let mut tx = self.pool.begin().await?;
        // An id is unique across all collections; refuse to move a row out of
        // another collection.
        for chunk in ids.chunks(ID_CHUNK) {
            let sql = format!(
                "SELECT id FROM {emb} WHERE collection_id <> ? AND id IN ({}) LIMIT 1 FOR UPDATE",
                placeholders(chunk.len())
            );
            let mut query = sqlx::query(AssertSqlSafe(sql)).bind(&self.collection_id);
            for id in chunk {
                query = query.bind(id);
            }
            if let Some(row) = query.fetch_optional(&mut *tx).await? {
                let id: String = row.try_get(0)?;
                return Err(
                    format!("document id `{id}` already belongs to another collection").into(),
                );
            }
        }
        for ((doc, vector), id) in docs.iter().zip(&vectors).zip(&ids) {
            sqlx::query(AssertSqlSafe(format!(
                "INSERT INTO {emb} (id, collection_id, embedding, document, cmetadata)
                 VALUES (?, ?, VEC_FromText(?), ?, ?)
                 ON DUPLICATE KEY UPDATE embedding = VALUE(embedding),
                     document = VALUE(document), cmetadata = VALUE(cmetadata)"
            )))
            .bind(id)
            .bind(&self.collection_id)
            .bind(vector_text(vector))
            .bind(&doc.page_content)
            .bind(serde_json::to_string(&doc.metadata)?)
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &Self::Options,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let filter = filter_json(opt)?;
        let threshold = score_threshold(opt)?;
        let collection = opt.name_space.as_deref().unwrap_or(&self.collection_name);
        if limit == 0 {
            return Ok(vec![]);
        }
        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);
        let query_vector = embedder.embed_query(query).await?;
        self.check_vector(&query_vector)?;

        let emb = &self.embedding_table_name;
        let coll = &self.collection_table_name;
        let filter_sql = if filter.is_some() {
            " AND JSON_CONTAINS(e.cmetadata, ?)"
        } else {
            ""
        };
        let threshold_sql = if threshold.is_some() {
            " HAVING distance <= ?"
        } else {
            ""
        };
        let sql = format!(
            "SELECT e.id, e.document, e.cmetadata,
                    VEC_DISTANCE_COSINE(e.embedding, VEC_FromText(?)) AS distance
             FROM {emb} e JOIN {coll} c ON e.collection_id = c.id
             WHERE c.name = ?{filter_sql}{threshold_sql}
             ORDER BY distance
             LIMIT ?"
        );
        let mut q = sqlx::query(AssertSqlSafe(sql))
            .bind(vector_text(&query_vector))
            .bind(collection);
        if let Some(filter) = &filter {
            q = q.bind(filter);
        }
        if let Some(threshold) = threshold {
            q = q.bind(1.0 - threshold);
        }
        let rows = q.bind(limit as u64).fetch_all(&self.pool).await?;
        rows.iter()
            .map(|row| {
                let distance: f64 = row.try_get("distance")?;
                Self::row_to_document(row, 1.0 - distance)
            })
            .collect()
    }

    /// Deletes this collection's documents with these ids; unknown ids are ignored.
    async fn delete(&self, ids: &[String], opt: &Self::Options) -> Result<(), Box<dyn Error>> {
        reject_query_options(opt, "delete")?;
        let emb = &self.embedding_table_name;
        for chunk in ids.chunks(ID_CHUNK) {
            let sql = format!(
                "DELETE FROM {emb} WHERE collection_id = ? AND id IN ({})",
                placeholders(chunk.len())
            );
            let mut query = sqlx::query(AssertSqlSafe(sql)).bind(&self.collection_id);
            for id in chunk {
                query = query.bind(id);
            }
            query.execute(&self.pool).await?;
        }
        Ok(())
    }

    /// This collection's documents with these ids, in request order (score 0). Unknown
    /// ids are skipped.
    async fn get_by_ids(
        &self,
        ids: &[String],
        opt: &Self::Options,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        reject_query_options(opt, "get_by_ids")?;
        let emb = &self.embedding_table_name;
        let mut docs = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(ID_CHUNK) {
            let sql = format!(
                "SELECT id, document, cmetadata FROM {emb}
                 WHERE collection_id = ? AND id IN ({})",
                placeholders(chunk.len())
            );
            let mut query = sqlx::query(AssertSqlSafe(sql)).bind(&self.collection_id);
            for id in chunk {
                query = query.bind(id);
            }
            for row in query.fetch_all(&self.pool).await? {
                docs.push(Self::row_to_document(&row, 0.0)?);
            }
        }
        Ok(in_id_order(ids, docs))
    }
}
