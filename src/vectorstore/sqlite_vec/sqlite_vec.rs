use std::{collections::HashMap, error::Error, sync::Arc};

use async_trait::async_trait;
use serde_json::{json, Value};
use sqlx::{Pool, Row, Sqlite};

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{VecStoreOptions, VectorStore},
};

pub struct Store {
    pub pool: Pool<Sqlite>,
    pub(crate) table: String,
    pub(crate) vector_dimensions: i32,
    pub(crate) embedder: Arc<dyn Embedder>,
}

impl Store {
    pub async fn initialize(&self) -> Result<(), Box<dyn Error>> {
        self.create_table_if_not_exists().await?;
        Ok(())
    }

    async fn create_table_if_not_exists(&self) -> Result<(), Box<dyn Error>> {
        let table = &self.table;

        sqlx::query(&format!(
            r#"
                CREATE TABLE IF NOT EXISTS {table}
                (
                  rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                  text TEXT,
                  metadata BLOB,
                  text_embedding BLOB
                )
                ;
                "#
        ))
        .execute(&self.pool)
        .await?;

        let dimensions = self.vector_dimensions;
        sqlx::query(&format!(
            r#"
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_{table} USING vec0(
                  text_embedding float[{dimensions}]
                );
                "#
        ))
        .execute(&self.pool)
        .await?;

        // NOTE: python langchain seems to only use "embed_text" as the trigger name
        sqlx::query(&format!(
            r#"
                CREATE TRIGGER IF NOT EXISTS embed_text_{table}
                AFTER INSERT ON {table}
                BEGIN
                    INSERT INTO vec_{table}(rowid, text_embedding)
                    VALUES (new.rowid, new.text_embedding)
                    ;
                END;
                "#
        ))
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    fn get_filters(&self, opt: &VecStoreOptions) -> Result<HashMap<String, Value>, Box<dyn Error>> {
        match &opt.filters {
            Some(Value::Object(map)) => {
                // Convert serde_json Map to HashMap<String, Value>
                let filters = map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                Ok(filters)
            }
            None => Ok(HashMap::new()), // No filters provided
            _ => Err("Invalid filters format".into()), // Filters provided but not in the expected format
        }
    }
}

#[async_trait]
impl VectorStore for Store {
    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &VecStoreOptions,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        let texts: Vec<String> = docs.iter().map(|d| d.page_content.clone()).collect();

        let embedder = opt.embedder.as_ref().unwrap_or(&self.embedder);

        let vectors = embedder.embed_documents(&texts).await?;
        if vectors.len() != docs.len() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Number of vectors and documents do not match",
            )));
        }

        let table = &self.table;

        let mut tx = self.pool.begin().await?;

        let mut ids = Vec::with_capacity(docs.len());

        for (doc, vector) in docs.iter().zip(vectors.iter()) {
            let text_embedding = json!(&vector);
            let id = sqlx::query(&format!(
                r#"
                    INSERT INTO {table}
                        (text, metadata, text_embedding)
                    VALUES
                        (?,?,?)"#
            ))
            .bind(&doc.page_content)
            .bind(json!(&doc.metadata))
            .bind(text_embedding.to_string())
            .execute(&mut *tx)
            .await?
            .last_insert_rowid();

            ids.push(id.to_string());
        }

        tx.commit().await?;

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &VecStoreOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let table = &self.table;

        let query_vector = json!(self.embedder.embed_query(query).await?);

        let filter = self.get_filters(opt)?;

        let mut metadata_query = filter
            .iter()
            .map(|(k, v)| format!("json_extract(e.metadata, '$.{}') = '{}'", k, v))
            .collect::<Vec<String>>()
            .join(" AND ");

        if metadata_query.is_empty() {
            metadata_query = "TRUE".to_string();
        }

        let rows = sqlx::query(&format!(
            r#"SELECT
                    text,
                    metadata,
                    distance
                FROM {table} e
                INNER JOIN vec_{table} v on v.rowid = e.rowid
                WHERE v.text_embedding match '{query_vector}' AND k = ? AND {metadata_query}
                ORDER BY distance
                LIMIT ?"#
        ))
        .bind(limit as i32)
        .bind(limit as i32)
        .fetch_all(&self.pool)
        .await?;

        let docs = rows
            .into_iter()
            .map(|row| {
                let page_content: String = row.try_get("text")?;
                let metadata_json: Value = row.try_get("metadata")?;
                let score: f64 = row.try_get("distance")?;

                let metadata = if let Value::Object(obj) = metadata_json {
                    obj.into_iter().collect()
                } else {
                    HashMap::new() // Or handle this case as needed
                };

                Ok(Document {
                    page_content,
                    metadata,
                    score,
                })
            })
            .collect::<Result<Vec<Document>, sqlx::Error>>()?;

        Ok(docs)
    }
}
