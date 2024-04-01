use async_trait::async_trait;

use crate::semantic_router::{IndexError, Router};

#[async_trait]
pub trait Index {
    async fn add(&mut self, router: &[Router]) -> Result<(), IndexError>;

    async fn delete(&mut self, route_name: &str) -> Result<(), IndexError>;

    async fn query(&self, vector: &[f64], top_k: usize) -> Result<Vec<(String, f64)>, IndexError>;

    async fn get_routes(&self) -> Result<Vec<Router>, IndexError>;

    async fn delete_index(&mut self) -> Result<(), IndexError>;
}
