use std::{collections::HashMap, error::Error, sync::Arc};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use surrealdb::{types::RecordId, Connection, Surreal};

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{vectorstore::in_id_order, VecStoreOptions, VectorStore},
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
                            DEFINE FIELD IF NOT EXISTS metadata                  ON {collection_table_name} TYPE option<object> FLEXIBLE;"#
                    ))
                    .await?
                    .check()?;
            }
            None => {
                let collection_table_name = &self.collection_name;
                self.db
                    .query(format!(
                        r#"
                            DEFINE TABLE IF NOT EXISTS {collection_table_name} SCHEMAFULL;
                            DEFINE FIELD IF NOT EXISTS text              ON {collection_table_name} TYPE string;
                            DEFINE FIELD IF NOT EXISTS embedding         ON {collection_table_name} TYPE array ASSERT (array::len($value) = {vector_dimensions}) || (array::len($value) = 0);
                            DEFINE FIELD IF NOT EXISTS embedding.*       ON {collection_table_name} TYPE float;
                            DEFINE FIELD IF NOT EXISTS metadata          ON {collection_table_name} TYPE option<object> FLEXIBLE;"#
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

    async fn add_documents(
        &self,
        docs: &[Document],
        opt: &Self::Options,
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

        let collection_predicate = self.collection_predicate();

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

        // surrealdb 3 decodes into `SurrealValue`; go through JSON to keep a plain serde `Row`.
        let rows: Vec<Value> = result.take(0)?;
        let query_result = rows
            .into_iter()
            .map(serde_json::from_value::<Row>)
            .collect::<Result<Vec<_>, _>>()?;

        let documents = query_result
            .into_iter()
            .map(|row| Document {
                page_content: row.text,
                metadata: row.metadata,
                score: row.similarity,
                id: Some(row.id),
            })
            .collect();

        Ok(documents)
    }

    /// Deletes records of this collection by id (the key `add_documents` returned).
    /// On a shared table only this collection's records are deleted. `filters` and
    /// `name_space` are rejected: they can't narrow a lookup by id here.
    async fn delete(&self, ids: &[String], opt: &Self::Options) -> Result<(), Box<dyn Error>> {
        reject_id_options(opt)?;
        if ids.is_empty() {
            return Ok(());
        }
        let collection_predicate = self.collection_predicate();
        self.db
            .query(format!("DELETE $records WHERE true {collection_predicate}"))
            .bind(("records", self.record_ids(ids)))
            .bind(("collection_name", self.collection_name.to_owned()))
            .bind(("collection_metadata_key", self.get_collection_metdata_key()))
            .await?
            .check()?;
        Ok(())
    }

    /// Fetches records of this collection by id, in request order (score 0).
    async fn get_by_ids(
        &self,
        ids: &[String],
        opt: &Self::Options,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        reject_id_options(opt)?;
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let collection_predicate = self.collection_predicate();
        let mut result = self
            .db
            .query(format!(
                "SELECT record::id(id) as id, text, metadata FROM $records \
                 WHERE true {collection_predicate}"
            ))
            .bind(("records", self.record_ids(ids)))
            .bind(("collection_name", self.collection_name.to_owned()))
            .bind(("collection_metadata_key", self.get_collection_metdata_key()))
            .await?
            .check()?;
        let rows: Vec<Value> = result.take(0)?;
        let docs = rows
            .into_iter()
            .map(serde_json::from_value::<StoredRow>)
            .map(|row| {
                row.map(|row| Document {
                    page_content: row.text,
                    metadata: row.metadata.unwrap_or_default(),
                    score: 0.0,
                    id: Some(row.id),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(in_id_order(ids, docs))
    }
}

impl<C: Connection> Store<C> {
    /// Restricts a query to this collection when it shares a table with others.
    fn collection_predicate(&self) -> &'static str {
        match &self.collection_table_name {
            Some(_) => " AND metadata[$collection_metadata_key] = $collection_name ",
            None => "",
        }
    }

    /// The records named by `ids` (keys in this store's table).
    fn record_ids(&self, ids: &[String]) -> Vec<RecordId> {
        let table = self.get_collection_table_name();
        ids.iter()
            .map(|id| RecordId::new(table, id.as_str()))
            .collect()
    }
}

fn reject_id_options(opt: &VecStoreOptions<Value>) -> Result<(), Box<dyn Error>> {
    if opt.filters.is_some() || opt.name_space.is_some() {
        return Err("surrealdb delete/get_by_ids don't support filters or name_space".into());
    }
    Ok(())
}

#[derive(Deserialize, Debug)]
struct StoredRow {
    id: String,
    text: String,
    metadata: Option<HashMap<String, Value>>,
}

#[derive(Deserialize, Debug)]
struct Row {
    id: String,
    text: String,
    metadata: HashMap<String, Value>,
    similarity: f64,
}
