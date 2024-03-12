use std::{error::Error, sync::Arc};

use surrealdb::{Connection, Surreal};

use crate::embedding::embedder_trait::Embedder;

use super::Store;

pub struct StoreBuilder<C: Connection> {
    db: Option<Surreal<C>>,
    collection_name: String,
    collection_table_name: Option<String>,
    collection_metadata_key_name: Option<String>,
    vector_dimensions: i32,
    embedder: Option<Arc<dyn Embedder>>,
    schemafull: bool,
}

impl<C: Connection> StoreBuilder<C> {
    /// Create a new StoreBuilder optimized for SurrealDB. Refer to `new_with_compatiblity()` if
    /// you are looking to connect to store created by python version of langchain.
    /// * table is singular - "document" instead of "documents"
    /// * uses single table instead of multiple tables
    /// * creates a schemafull table required for faster indexing. https://github.com/surrealdb/surrealdb/issues/2013
    pub fn new() -> Self {
        StoreBuilder {
            db: None,
            collection_name: "document".to_string(),
            collection_table_name: Some("document".to_string()),
            collection_metadata_key_name: Some("collection".to_string()),
            vector_dimensions: 0,
            embedder: None,
            schemafull: true,
        }
    }

    /// Create a new StoreBuilder with compatibility with python version of langchain
    pub fn new_with_compatiblity() -> Self {
        StoreBuilder {
            db: None,
            collection_name: "documents".to_string(),
            collection_table_name: None,
            collection_metadata_key_name: None,
            vector_dimensions: 0,
            embedder: None,
            schemafull: false,
        }
    }

    /// Use surrealdb
    /// ```no_run
    /// let surrealdb_config = surrealdb::opt::Config::new()
    ///     .set_strict(true)
    ///     .capabilities(Capabilities::all())
    ///     .user(surrealdb::opt::auth::Root {
    ///         username: "username".into(),
    ///         password: "password".into()
    ///     });
    /// let db = surrealdb::engine::any::connect(("ws://127.0.0.1:8000", surrealdb_config)).await?;
    /// let store = StoreBuilder::new().db(db).vector_dimensions(1000).build()?;
    /// store.initialize().await?;
    /// ```
    pub fn db(mut self, db: Surreal<C>) -> Self {
        self.db = Some(db);
        self
    }

    pub fn collection_name(mut self, collection_name: &str) -> Self {
        self.collection_name = collection_name.into();
        self
    }

    /// Setting collection_table_name to None, creates table per collection. Set to some value if
    /// you would like to reuse table. Resuing table is not compatible with python version of
    /// langchain.
    pub fn collection_table_name(mut self, collection_table_name: Option<String>) -> Self {
        self.collection_table_name = collection_table_name;
        self
    }

    pub fn collection_metadata_key_name(
        mut self,
        collection_metadata_key_name: Option<String>,
    ) -> Self {
        self.collection_metadata_key_name = collection_metadata_key_name;
        self
    }

    pub fn vector_dimensions(mut self, vector_dimensions: i32) -> Self {
        self.vector_dimensions = vector_dimensions;
        self
    }

    pub fn schemafull(mut self, schemafull: bool) -> Self {
        self.schemafull = schemafull;
        self
    }

    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    // Finalize the builder and construct the Store object
    pub async fn build(self) -> Result<Store<C>, Box<dyn Error>> {
        if self.embedder.is_none() {
            return Err("Embedder is required".into());
        }

        if self.db.is_none() {
            return Err("Db is required".into());
        }

        Ok(Store {
            db: self.db.unwrap(),
            collection_name: self.collection_name,
            collection_table_name: self.collection_table_name,
            collection_metadata_key_name: self.collection_metadata_key_name,
            vector_dimensions: self.vector_dimensions,
            embedder: self.embedder.unwrap(),
            schemafull: self.schemafull,
        })
    }
}
