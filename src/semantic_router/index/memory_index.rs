use std::collections::HashMap;

use async_trait::async_trait;

use crate::semantic_router::{utils::cosine_similarity, IndexError, Router};

use super::Index;

pub struct MemoryIndex {
    routers: HashMap<String, Router>,
}
impl MemoryIndex {
    pub fn new() -> Self {
        return Self {
            routers: HashMap::new(),
        };
    }
}

#[async_trait]
impl Index for MemoryIndex {
    async fn add(&mut self, routers: &[Router]) -> Result<(), IndexError> {
        for router in routers {
            if router.embedding.is_none() {
                return Err(IndexError::MissingEmbedding(router.name.clone()));
            }
            if self.routers.contains_key(&router.name) {
                log::warn!("Router {} already exists in the index", router.name);
            }
            self.routers.insert(router.name.clone(), router.clone());
        }

        Ok(())
    }

    async fn delete(&mut self, router_name: &str) -> Result<(), IndexError> {
        if self.routers.remove(router_name).is_none() {
            log::warn!("Router {} not found in the index", router_name);
        }
        Ok(())
    }

    async fn query(&self, vector: &[f64], top_k: usize) -> Result<Vec<(String, f64)>, IndexError> {
        let mut all_similarities: Vec<(String, f64)> = Vec::new();

        // Compute similarity for each embedding of each router
        for (name, router) in &self.routers {
            if let Some(embeddings) = &router.embedding {
                for embedding in embeddings {
                    let similarity = cosine_similarity(vector, embedding);
                    all_similarities.push((name.clone(), similarity));
                }
            }
        }

        // Sort all similarities by descending similarity score
        all_similarities
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Only keep the top_k similarities
        let top_similarities: Vec<(String, f64)> =
            all_similarities.into_iter().take(top_k).collect();

        Ok(top_similarities)
    }

    async fn get_routers(&self) -> Result<Vec<Router>, IndexError> {
        let routes = self.routers.values().cloned().collect();
        Ok(routes)
    }

    async fn get_router(&self, route_name: &str) -> Result<Router, IndexError> {
        return self
            .routers
            .get(route_name)
            .cloned()
            .ok_or(IndexError::RouterNotFound(route_name.into()));
    }

    async fn delete_index(&mut self) -> Result<(), IndexError> {
        self.routers.clear();
        Ok(())
    }
}
