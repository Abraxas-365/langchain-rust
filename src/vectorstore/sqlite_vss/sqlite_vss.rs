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
    pub(crate) pool: Pool<Sqlite>,
    pub(crate) table: String,
    pub(crate) vector_dimensions: i32,
    pub(crate) embedder: Arc<dyn Embedder>,
}

pub type SqliteVssOptions = VecStoreOptions<Value>;

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
                CREATE VIRTUAL TABLE IF NOT EXISTS vss_{table} USING vss0(
                  text_embedding({dimensions})
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
                    INSERT INTO vss_{table}(rowid, text_embedding)
                    VALUES (new.rowid, new.text_embedding)
                    ;
                END;
                "#
        ))
        .execute(&self.pool)
        .await?;

        Ok(())
    }
}

#[async_trait]
impl VectorStore for Store {
    type Options = SqliteVssOptions;

    async fn add_documents(
        &self,
        docs: &[Document],
        opt: Option<&Self::Options>,
    ) -> Result<Vec<String>, Box<dyn Error>> {
         let embedder = if let Some(options) = opt {
            options.embedder.as_ref().unwrap_or(&self.embedder)
        } else {
            &self.embedder
        };

        let texts: Vec<String> = docs.iter().map(|d| d.page_content.clone()).collect();

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
        _opt: &Self::Options,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let table = &self.table;

        let query_vector = json!(self.embedder.embed_query(query).await?);

        let rows = sqlx::query(&format!(
            r#"SELECT
                    text,
                    metadata,
                    distance
                FROM {table} e
                INNER JOIN vss_{table} v on v.rowid = e.rowid
                WHERE vss_search(
                  v.text_embedding,
                  vss_search_params('{query_vector}', ?)
                )
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
