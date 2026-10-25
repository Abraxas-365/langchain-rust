use std::{collections::HashMap, error::Error, sync::Arc};

use serde_json::Value;
use sqlx::{mysql::MySqlPoolOptions, AssertSqlSafe, MySql, Pool, Row};
use uuid::Uuid;

use crate::embedding::embedder_trait::Embedder;

use super::{check_identifier, Store};

const DEFAULT_COLLECTION_NAME: &str = "langchain";
const DEFAULT_EMBEDDING_TABLE_NAME: &str = "langchain_mariadb_embedding";
const DEFAULT_COLLECTION_TABLE_NAME: &str = "langchain_mariadb_collection";
/// Longest collection name the `varchar` column holds.
const MAX_COLLECTION_NAME_CHARS: usize = 255;

/// Builds a MariaDB [`Store`]. Needs MariaDB 11.7 or newer (native `VECTOR` type).
///
/// ```rust,ignore
/// let store = StoreBuilder::new()
///     .embedder(OpenAiEmbedder::default())
///     .connection_url("mysql://user:pw@localhost:3306/mydb")
///     .vector_dimensions(1536)
///     .collection_name("docs")
///     .build()
///     .await?;
/// ```
///
/// On build it creates the collection and embedding tables if they are missing, plus a
/// row for the collection. Several collections share the same two tables.
pub struct StoreBuilder {
    pool: Option<Pool<MySql>>,
    connection_url: Option<String>,
    embedder: Option<Arc<dyn Embedder>>,
    vector_dimensions: usize,
    collection_name: String,
    collection_metadata: HashMap<String, Value>,
    collection_table_name: String,
    embedding_table_name: String,
    pre_delete_collection: bool,
}

impl Default for StoreBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl StoreBuilder {
    pub fn new() -> Self {
        StoreBuilder {
            pool: None,
            connection_url: None,
            embedder: None,
            vector_dimensions: 0,
            collection_name: DEFAULT_COLLECTION_NAME.into(),
            collection_metadata: HashMap::new(),
            collection_table_name: DEFAULT_COLLECTION_TABLE_NAME.into(),
            embedding_table_name: DEFAULT_EMBEDDING_TABLE_NAME.into(),
            pre_delete_collection: false,
        }
    }

    /// Use an existing pool instead of `connection_url`.
    pub fn pool(mut self, pool: Pool<MySql>) -> Self {
        self.pool = Some(pool);
        self.connection_url = None;
        self
    }

    /// `mysql://user:password@host:3306/database`.
    pub fn connection_url<S: Into<String>>(mut self, connection_url: S) -> Self {
        self.connection_url = Some(connection_url.into());
        self.pool = None;
        self
    }

    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    /// Size of the embedder's vectors. Required; fixed when the table is created.
    pub fn vector_dimensions(mut self, vector_dimensions: usize) -> Self {
        self.vector_dimensions = vector_dimensions;
        self
    }

    /// Collection the store reads and writes (default `langchain`).
    pub fn collection_name<S: Into<String>>(mut self, collection_name: S) -> Self {
        self.collection_name = collection_name.into();
        self
    }

    /// Stored on the collection row; replaced on every build.
    pub fn collection_metadata(mut self, collection_metadata: HashMap<String, Value>) -> Self {
        self.collection_metadata = collection_metadata;
        self
    }

    /// Letters, digits and `_` only (default `langchain_mariadb_collection`).
    pub fn collection_table_name<S: Into<String>>(mut self, name: S) -> Self {
        self.collection_table_name = name.into();
        self
    }

    /// Letters, digits and `_` only (default `langchain_mariadb_embedding`).
    pub fn embedding_table_name<S: Into<String>>(mut self, name: S) -> Self {
        self.embedding_table_name = name.into();
        self
    }

    /// Delete the collection (and, through the foreign key, its documents) before
    /// recreating it empty.
    pub fn pre_delete_collection(mut self, pre_delete_collection: bool) -> Self {
        self.pre_delete_collection = pre_delete_collection;
        self
    }

    pub async fn build(self) -> Result<Store, Box<dyn Error>> {
        let embedder = self.embedder.clone().ok_or("embedder is required")?;
        if self.vector_dimensions == 0 {
            return Err("vector_dimensions must be greater than zero".into());
        }
        check_identifier("collection_table_name", &self.collection_table_name)?;
        check_identifier("embedding_table_name", &self.embedding_table_name)?;
        if self.collection_table_name == self.embedding_table_name {
            return Err("collection_table_name and embedding_table_name must differ".into());
        }
        if self.collection_name.is_empty()
            || self.collection_name.chars().count() > MAX_COLLECTION_NAME_CHARS
        {
            return Err(format!(
                "collection_name must be 1 to {MAX_COLLECTION_NAME_CHARS} characters"
            )
            .into());
        }
        let pool = self.get_pool().await?;

        let mut store = Store {
            pool,
            embedder,
            vector_dimensions: self.vector_dimensions,
            collection_name: self.collection_name,
            collection_id: String::new(),
            collection_table_name: self.collection_table_name,
            embedding_table_name: self.embedding_table_name,
        };
        store.create_tables().await?;
        store.collection_id = create_or_get_collection(
            &store,
            &self.collection_metadata,
            self.pre_delete_collection,
        )
        .await?;
        Ok(store)
    }

    async fn get_pool(&self) -> Result<Pool<MySql>, Box<dyn Error>> {
        match (&self.pool, &self.connection_url) {
            (Some(pool), _) => Ok(pool.clone()),
            (None, Some(url)) => Ok(MySqlPoolOptions::new().connect(url).await?),
            (None, None) => Err("connection_url or pool is required".into()),
        }
    }
}

/// Finds or creates the collection row and returns its id. The unique name makes this
/// safe when several processes build the same store at once.
async fn create_or_get_collection(
    store: &Store,
    metadata: &HashMap<String, Value>,
    pre_delete: bool,
) -> Result<String, Box<dyn Error>> {
    let table = &store.collection_table_name;
    let metadata = serde_json::to_string(metadata)?;
    let mut tx = store.pool.begin().await?;
    if pre_delete {
        sqlx::query(AssertSqlSafe(format!("DELETE FROM {table} WHERE name = ?")))
            .bind(&store.collection_name)
            .execute(&mut *tx)
            .await?;
    }
    sqlx::query(AssertSqlSafe(format!(
        "INSERT IGNORE INTO {table} (id, name, cmetadata) VALUES (?, ?, ?)"
    )))
    .bind(Uuid::new_v4().to_string())
    .bind(&store.collection_name)
    .bind(&metadata)
    .execute(&mut *tx)
    .await?;
    let id: String = sqlx::query(AssertSqlSafe(format!(
        "SELECT id FROM {table} WHERE name = ?"
    )))
    .bind(&store.collection_name)
    .fetch_one(&mut *tx)
    .await?
    .try_get(0)?;
    sqlx::query(AssertSqlSafe(format!(
        "UPDATE {table} SET cmetadata = ? WHERE id = ?"
    )))
    .bind(&metadata)
    .bind(&id)
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(id)
}
