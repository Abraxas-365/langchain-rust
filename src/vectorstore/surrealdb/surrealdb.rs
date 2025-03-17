use std::{collections::HashMap, error::Error, sync::Arc};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use surrealdb::{Connection, Surreal};

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{VecStoreOptions, VectorStore},
};

// INSERT INTO documents {
//  text: 'some text,
//  embedding: [1.0, 2.0, 3.0],
//  metadata?: {},
//  collection?: 'collection_name'
// }

pub struct Store<C: Connection> {
    pub(crate) db: Surreal<C>,
    pub(crate) collection_name: String,
    pub(crate) collection_table_name: Option<String>,
    pub(crate) collection_metadata_key_name: Option<String>,
    pub(crate) vector_dimensions: i32,
    pub(crate) embedder: Arc<dyn Embedder>,
    pub(crate) schemafull: bool,
}

impl<C: Connection> Store<C> {
    fn get_collection_table_name(&self) -> &str {
        match &self.collection_table_name {
            Some(collection_table_name) => collection_table_name.as_str(),
            None => self.collection_name.as_str(),
        }
    }

    fn get_collection_metdata_key(&self) -> String {
        self.collection_metadata_key_name
            .clone()
            .unwrap_or_else(|| "collection".to_string())
    }

    pub async fn initialize(&self) -> Result<(), Box<dyn Error>> {
        self.create_collection_table_if_not_exists().await?;
        Ok(())
    }

    async fn create_collection_table_if_not_exists(&self) -> Result<(), Box<dyn Error>> {
        if !self.schemafull {
            return Ok(());
        }

        let vector_dimensions = self.vector_dimensions;

        match &self.collection_table_name {
            Some(collection_table_name) => {
                self.db
                    .query(format!(
                        r#"
                            DEFINE TABLE IF NOT EXISTS {collection_table_name} SCHEMAFULL;
                            DEFINE FIELD IF NOT EXISTS text                      ON {collection_table_name} TYPE string;
                            DEFINE FIELD IF NOT EXISTS embedding                 ON {collection_table_name} TYPE array ASSERT (array::len($value) = {vector_dimensions}) || (array::len($value) = 0);
                            DEFINE FIELD IF NOT EXISTS embedding.*               ON {collection_table_name} TYPE float;
                            DEFINE FIELD IF NOT EXISTS metadata                  ON {collection_table_name} FLEXIBLE TYPE option<object>;"#
                    ))
                    .await?
                    .check()?;
            }
            None => {
                let collection_table_name = &self.collection_name;
                dbg!(&collection_table_name);
                self.db
                    .query(format!(
                        r#"
                            DEFINE TABLE IF NOT EXISTS {collection_table_name} SCHEMAFULL;
                            DEFINE FIELD IF NOT EXISTS text              ON {collection_table_name} TYPE string;
                            DEFINE FIELD IF NOT EXISTS embedding         ON {collection_table_name} TYPE array ASSERT (array::len($value) = {vector_dimensions}) || (array::len($value) = 0);
                            DEFINE FIELD IF NOT EXISTS embedding.*       ON {collection_table_name} TYPE float;
                            DEFINE FIELD IF NOT EXISTS metadata          ON {collection_table_name} FLEXIBLE TYPE option<object>;"#
                    ))
                    .await?
                    .check()?;
            }
        }

        Ok(())
    }
}

#[async_trait]
impl<C: Connection> VectorStore for Store<C> {
    type Options = VecStoreOptions<Value>;

    /// Only uses `embedder` from passed options.
    /// (Defaults to the connected `VectorStore`'s previously configured embedder.)
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

        let mut ids = Vec::with_capacity(docs.len());

        for (doc, vector) in docs.iter().zip(vectors.iter()) {
            match &self.collection_table_name {
                Some(collection_table_name) => {
                    let mut metadata: HashMap<String, Value> = doc.metadata.clone();
                    metadata.insert(
                        self.get_collection_metdata_key(),
                        Value::String(self.collection_name.to_owned()),
                    );

                    let mut result = self
                        .db
                        .query(format!(
                            r#"CREATE {collection_table_name} CONTENT {{
                                text: $text,
                                embedding: $embedding,
                                metadata: $metadata,
                            }}
                            RETURN record::id(id) as id"#
                        ))
                        .bind(("text", doc.page_content.to_owned()))
                        .bind(("embedding", vector.to_owned()))
                        .bind(("metadata", metadata.to_owned()))
                        .await?
                        .check()?;

                    let id: Option<String> = result.take("id")?;
                    ids.push(id.unwrap());
                }
                None => {
                    let collection_table_name = &self.collection_name;
                    let mut result = self
                        .db
                        .query(format!(
                            r#"CREATE {collection_table_name} CONTENT {{
                                text: $text,
                                embedding: $embedding,
                                metadata: $metadata,
                            }}
                            RETURN record::id(id) as id"#
                        ))
                        .bind(("text", doc.page_content.to_owned()))
                        .bind(("embedding", vector.to_owned()))
                        .bind(("metadata", doc.metadata.to_owned()))
                        .await?
                        .check()?;

                    let id: Option<String> = result.take("id")?;
                    ids.push(id.unwrap());
                }
            }
        }

        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query: &str,
        limit: usize,
        opt: &Self::Options,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let collection_name = &self.collection_name;
        let collection_table_name = self.get_collection_table_name();

        let query_vector = self.embedder.embed_query(query).await?;

        let collection_predicate = match &self.collection_table_name {
            Some(_) => " AND metadata[$collection_metadata_key] = $collection_name ",
            None => "",
        };

        let mut result = self
            .db
            .query(format!(
                r#"
        SELECT record::id(id) as id, text, metadata,
        vector::similarity::cosine(embedding, $embedding) as similarity
        FROM {collection_table_name}
        WHERE vector::similarity::cosine(embedding, $embedding) >= $score_threshold {collection_predicate}
        ORDER BY similarity DESC LIMIT $k
            "#
            ))
            .bind(("collection_name", collection_name.to_owned()))
            .bind(("collection_metadata_key", self.get_collection_metdata_key().to_owned()))
            .bind(("score_threshold", opt.score_threshold.unwrap_or(0.0)))
            .bind(("k", limit))
            .bind(("embedding", query_vector.to_owned()))
            .await?
            .check()?;

        let query_result: Vec<Row> = result.take(0)?;

        let documents = query_result
            .into_iter()
            .map(|row| Document {
                page_content: row.text,
                metadata: row.metadata,
                score: row.similarity,
            })
            .collect();

        Ok(documents)
    }
}

#[derive(Deserialize, Debug)]
struct Row {
    id: String,
    text: String,
    metadata: HashMap<String, Value>,
    similarity: f64,
}
