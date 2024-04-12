use crate::embedding::Embedder;
use crate::vectorstore::opensearch::Store;
use aws_config::SdkConfig;
use opensearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use opensearch::OpenSearch;
use std::error::Error;
use std::sync::Arc;
use url::Url;

pub struct StoreBuilder {
    client: Option<OpenSearch>,
    embedder: Option<Arc<dyn Embedder>>,
    k: i32,
    index: Option<String>,
    vector_field: String,
    content_field: String,
}

impl StoreBuilder {
    // Returns a new StoreBuilder instance with default values for each option
    pub fn new() -> Self {
        StoreBuilder {
            client: None,
            embedder: None,
            k: 2,
            index: None,
            vector_field: "vector_field".to_string(),
            content_field: "page_content".to_string(),
        }
    }

    pub fn client(mut self, client: OpenSearch) -> Self {
        self.client = Some(client);
        self
    }

    pub fn aoss_client(
        mut self,
        sdk_config: &SdkConfig,
        host: &str,
    ) -> Result<Self, Box<dyn Error>> {
        let client = build_aoss_client(sdk_config, host)?;
        self.client = Some(client);
        Ok(self)
    }

    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    pub fn k(mut self, k: i32) -> Self {
        self.k = k;
        self
    }

    pub fn index(mut self, index: &str) -> Self {
        self.index = Some(index.to_string());
        self
    }

    pub fn vector_field(mut self, vector_field: &str) -> Self {
        self.vector_field = vector_field.to_string();
        self
    }

    pub fn content_field(mut self, content_field: &str) -> Self {
        self.content_field = content_field.to_string();
        self
    }

    // Finalize the builder and construct the Store object
    pub async fn build(self) -> Result<Store, Box<dyn Error>> {
        if self.client.is_none() {
            return Err("Client is required".into());
        }

        if self.embedder.is_none() {
            return Err("Embedder is required".into());
        }

        if self.index.is_none() {
            return Err("Index is required".into());
        }

        Ok(Store {
            client: self.client.unwrap(),
            embedder: self.embedder.unwrap(),
            k: self.k,
            index: self.index.unwrap(),
            vector_field: self.vector_field,
            content_field: self.content_field,
        })
    }
}

fn build_aoss_client(sdk_config: &SdkConfig, host: &str) -> Result<OpenSearch, Box<dyn Error>> {
    let opensearch_url = Url::parse(host)?;
    let conn_pool = SingleNodeConnectionPool::new(opensearch_url);

    let transport = TransportBuilder::new(conn_pool)
        .auth(sdk_config.try_into()?)
        .service_name("aoss")
        .build()?;
    let client = OpenSearch::new(transport);
    Ok(client)
}
