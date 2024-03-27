use async_trait::async_trait;

use crate::semantic_router::{IndexError, Router};

#[async_trait]
pub trait Index {
    async fn add(&self, router: &[Router]) -> Result<(), IndexError>;

    async fn delete(&self, route_name: &str) -> Result<(), IndexError>;

    async fn query(&self, vector: &[f64], top_k: usize) -> Result<Vec<Router>, IndexError>;

    async fn get_routes(&self) -> Result<Vec<Router>, IndexError>;

    async fn delete_index(&self) -> Result<(), IndexError>;
}
