use std::{collections::HashMap, error::Error, sync::Arc};

use async_trait::async_trait;
use pgvector::Vector;
use serde_json::{json, Value};
use sqlx::{Pool, Postgres, Row};
use uuid::Uuid;

use crate::{
    embedding::embedder_trait::Embedder,
    schemas::Document,
    vectorstore::{VecStoreOptions, VectorStore},
};

pub struct Store {
    pub(crate) embedder: Arc<dyn Embedder>,
    pub(crate) pool: Pool<Postgres>,
    pub(crate) collection_name: String,
    pub(crate) collection_table_name: String,
    pub(crate) collection_uuid: String,
    pub(crate) collection_metadata: HashMap<String, Value>,
    pub(crate) embedder_table_name: String,
    pub(crate) pre_delete_collection: bool,
    pub(crate) vector_dimensions: i32,
    pub(crate) hns_index: Option<HNSWIndex>,
    pub(crate) vstore_options: PgOptions,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PgFilter {
    Eq(PgLit, PgLit),
    Cmp(std::cmp::Ordering, PgLit, PgLit),
    In(PgLit, Vec<String>),
    And(Vec<PgFilter>),
    Or(Vec<PgFilter>),
}

pub type Column = String;

pub type Path = Vec<String>;

#[derive(Debug, Clone, PartialEq)]
pub enum PgLit {
    JsonField(Path),
    LitStr(String),
    RawJson(Value),
}

impl ToString for PgLit {
    fn to_string(&self) -> String {
        match self {
            PgLit::LitStr(str) => format!("'{}'", str.clone()),
            PgLit::JsonField(path) => format!("cmetadata#>>'{{{}}}'", path.join(",")),
            PgLit::RawJson(value) => serde_json::to_string(value).unwrap_or("null".to_string()),
        }
    }
}

impl ToString for PgFilter {
    fn to_string(&self) -> String {
        match self {
            PgFilter::Eq(a, b) => format!("{} = {}", a.to_string(), b.to_string()),
            PgFilter::Cmp(ordering, a, b) => {
                let op = match ordering {
                    std::cmp::Ordering::Less => "<",
                    std::cmp::Ordering::Greater => ">",
                    std::cmp::Ordering::Equal => "=",
                };
                format!("{} {} {}", a.to_string(), op, b.to_string())
            }
            PgFilter::In(a, values) => {
                format!(
                    "{} = ANY(ARRAY[{}])",
                    a.to_string(),
                    values
                        .iter()
                        .map(|s| format!("'{}'", s))
                        .collect::<Vec<String>>()
                        .join(",")
                )
            }
            PgFilter::And(pgfilters) => pgfilters
                .iter()
                .map(|pgf| pgf.to_string())
                .collect::<Vec<String>>()
                .join(" AND "),
            PgFilter::Or(pgfilters) => pgfilters
                .iter()
                .map(|pgf| pgf.to_string())
                .collect::<Vec<String>>()
                .join(" OR "),
        }
    }
}

pub struct HNSWIndex {
    pub(crate) m: i32,
    pub(crate) ef_construction: i32,
    pub(crate) distance_function: String,
}

impl HNSWIndex {
    pub fn new(m: i32, ef_construction: i32, distance_function: &str) -> Self {
        HNSWIndex {
            m,
            ef_construction,
            distance_function: distance_function.into(),
        }
    }
}

impl Store {
    fn get_filters(&self, opt: &PgOptions) -> Result<String, Box<dyn Error>> {
        match &opt.filters {
            Some(pgfilter) => Ok(pgfilter.to_string()),
            None => Ok("TRUE".to_string()), // No filters provided
        }
    }

    fn get_name_space(&self, opt: &PgOptions) -> String {
        match &opt.name_space {
            Some(name_space) => name_space.clone(),
            None => self.collection_name.clone(),
        }
    }

    fn get_score_threshold(&self, opt: &PgOptions) -> Result<f32, Box<dyn Error>> {
        match &opt.score_threshold {
            Some(score_threshold) => {
                if *score_threshold < 0.0 || *score_threshold > 1.0 {
                    return Err("Invalid score threshold".into());
                }
                Ok(*score_threshold)
            }
            None => Ok(0.0),
        }
    }

    async fn drop_tables(&self) -> Result<(), Box<dyn Error>> {
        sqlx::query(&format!(
            r#"DROP TABLE IF EXISTS {}"#,
            self.embedder_table_name
        ))
        .execute(&self.pool)
        .await?;

        sqlx::query(&format!(
            r#"DROP TABLE IF EXISTS {}"#,
            self.collection_table_name
        ))
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    async fn remove_collection(&self) -> Result<(), Box<dyn Error>> {
        sqlx::query(r#"DELETE FROM collection WHERE uuid = $1"#)
            .bind(&self.collection_uuid)
            .execute(&self.pool)
            .await?;
        Ok(())
    }
}

pub type PgOptions = VecStoreOptions<PgFilter>;

impl Default for PgOptions {
    fn default() -> Self {
        PgOptions {
            filters: None,
            score_threshold: None,
            name_space: None,
            embedder: None,
        }
    }
}

#[async_trait]
impl VectorStore for Store {
    type Options = PgOptions;

    async fn add_documents(
        &self,
        docs: &[Document],
        opt: Option<&PgOptions>,
    ) -> Result<Vec<String>, Box<dyn Error>> {
        if let Some(option) = opt {
            if option.score_threshold.is_some() || option.filters.is_some() || option.name_space.is_some() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "score_threshold, filters, and name_space are not supported in pgvector",
                )));
            }
        }

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

        let mut tx = self.pool.begin().await?;

        let mut ids = Vec::with_capacity(docs.len());

        for (doc, vector) in docs.iter().zip(vectors.iter()) {
            let id = Uuid::new_v4().to_string();
            ids.push(id.clone());

            let vector_value =
                Vector::from(vector.into_iter().map(|x| *x as f32).collect::<Vec<f32>>());

            sqlx::query(&format!(
                r#"INSERT INTO {} 
(uuid, document, embedding, cmetadata, collection_id) VALUES ($1, $2, $3, $4, $5)"#,
                self.embedder_table_name
            ))
            .bind(&id)
            .bind(&doc.page_content)
            .bind(&vector_value)
            .bind(json!(&doc.metadata))
            .bind(&self.collection_uuid)
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
        opt: &PgOptions,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let collection_name = self.get_name_space(opt);
        let where_filter = self.get_filters(opt)?;

        let sql = format!(
            r#"WITH filtered_embedding_dims AS MATERIALIZED (
                SELECT
                    *
                FROM
                    {}
                WHERE
                    vector_dims(embedding) = $1
            )
            SELECT
                data.document,
                data.cmetadata,
                data.distance
            FROM (
                SELECT
                    filtered_embedding_dims.*,
                    embedding <=> $2 AS distance
                FROM
                    filtered_embedding_dims
                    JOIN {} ON filtered_embedding_dims.collection_id = {}.uuid
                WHERE {}.name = '{}'
            ) AS data
            WHERE {}
            ORDER BY
                data.distance DESC
            LIMIT $3"#,
            self.embedder_table_name,
            self.collection_table_name,
            self.collection_table_name,
            self.collection_table_name,
            collection_name,
            where_filter,
        );

        let query_vector = self.embedder.embed_query(query).await?;

        let vector_dims = query_vector.len();

        let rows = sqlx::query(&sql)
            .bind(vector_dims as i64)
            .bind(&Vector::from(
                query_vector
                    .into_iter()
                    .map(|x| x as f32)
                    .collect::<Vec<f32>>(),
            ))
            .bind(limit as i32)
            .fetch_all(&self.pool)
            .await?;

        let docs = rows
            .into_iter()
            .map(|row| {
                let page_content: String = row.try_get(0)?;
                let metadata_json: Value = row.try_get(1)?;
                let score: f64 = row.try_get(2)?;

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
