use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::semantic_router::{IndexError, Router};

use super::Index;

pub struct LocalIndex {
    routers: Mutex<HashMap<String, Router>>,
}
impl LocalIndex {
    pub fn new() -> Self {
        return Self {
            routers: Mutex::new(HashMap::new()),
        };
    }
}

#[async_trait]
impl Index for LocalIndex {
    async fn add(&self, routers: &[Router]) -> Result<(), IndexError> {
        let mut locked_routers = self.routers.lock().await;
        for router in routers {
            if router.embedding.is_none() {
                return Err(IndexError::MissingEmbedding(router.name.clone()));
            }
            if locked_routers.contains_key(&router.name) {
                log::warn!("Router {} already exists in the index", router.name);
            }
            locked_routers.insert(router.name.clone(), router.clone());
        }

        Ok(())
    }

    async fn delete(&self, router_name: &str) -> Result<(), IndexError> {
        let mut locked_routers = self.routers.lock().await;
        if locked_routers.remove(router_name).is_none() {
            log::warn!("Router {} not found in the index", router_name);
        }
        Ok(())
    }

    async fn query(&self, vector: &[f64], top_k: usize) -> Result<Vec<Router>, IndexError> {
        todo!()
    }

    async fn get_routes(&self) -> Result<Vec<Router>, IndexError> {
        let routes = self.routers.lock().await.values().cloned().collect();
        Ok(routes)
    }

    async fn delete_index(&self) -> Result<(), IndexError> {
        let mut locked_routers = self.routers.lock().await;
        locked_routers.clear();
        Ok(())
    }
}
