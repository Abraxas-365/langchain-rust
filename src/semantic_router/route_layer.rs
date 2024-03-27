use std::sync::Arc;

use crate::{
    embedding::{openai::OpenAiEmbedder, Embedder},
    language_models::llm::LLM,
    llm::openai::OpenAI,
};

use super::{
    utils::combine_embeddings, Index, LocalIndex, RouteLayerBuilderError, RouteLayerError, Router,
};

/// A builder for creating a `RouteLayer`.
///```rust,ignore
/// let politics_route = RouterBuilder::new("politics")
///     .utterances(&[
///         "isn't politics the best thing ever",
///         "why don't you tell me about your political opinions",
///         "don't you just love the president",
///         "they're going to destroy this country!",
///         "they will save the country!",
///     ])
///     .build()
///     .unwrap();
///
/// let chitchat_route = RouterBuilder::new("chitchat")
///     .utterances(&[
///         "how's the weather today?",
///         "how are things going?",
///         "lovely weather today",
///         "the weather is horrendous",
///         "let's go to the chippy",
///     ])
///     .build()
///     .unwrap();
///
/// let router_layer = RouteLayerBuilder::new()
///     .embedder(OpenAiEmbedder::default())
///     .add_route(politics_route)
///     .add_route(chitchat_route)
///     .threshold(0.7)
///     .build()
///     .await
///     .unwrap();
/// ```
///
pub struct RouteLayerBuilder {
    embedder: Option<Arc<dyn Embedder>>,
    routes: Vec<Router>,
    threshold: Option<f64>,
    index: Option<Arc<dyn Index>>,
    llm: Option<Arc<dyn LLM>>,
}
impl Default for RouteLayerBuilder {
    fn default() -> Self {
        Self::new()
            .embedder(OpenAiEmbedder::default())
            .llm(OpenAI::default())
            .index(LocalIndex::new())
    }
}

impl RouteLayerBuilder {
    pub fn new() -> Self {
        Self {
            embedder: None,
            routes: Vec::new(),
            threshold: None,
            llm: None,
            index: None,
        }
    }

    pub fn llm<L: LLM + 'static>(mut self, llm: L) -> Self {
        self.llm = Some(Arc::new(llm));
        self
    }

    pub fn index<I: Index + 'static>(mut self, index: I) -> Self {
        self.index = Some(Arc::new(index));
        self
    }

    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn add_route(mut self, route: Router) -> Self {
        self.routes.push(route);
        self
    }

    pub async fn build(mut self) -> Result<RouteLayer, RouteLayerBuilderError> {
        // Check if any routers lack an embedding and there's no global embedder provided.
        if self.embedder.is_none() {
            return Err(RouteLayerBuilderError::MissingEmbedder);
        }

        if self.llm.is_none() {
            return Err(RouteLayerBuilderError::MissingLLM);
        }

        if self.index.is_none() {
            return Err(RouteLayerBuilderError::MissingIndex);
        }

        let router = RouteLayer {
            embedder: self.embedder.unwrap(), //it's safe to unwrap here because we checked for None above
            index: self.index.unwrap(),
            llm: self.llm.unwrap(),
            threshold: self.threshold.unwrap_or(0.7),
        };
        for route in self.routes.iter_mut() {
            if route.embedding.is_none() {
                let embeddings = router.embedder.embed_documents(&route.utterances).await?;
                route.embedding = Some(embeddings);
            }
        }
        router.index.add(&self.routes).await?;

        Ok(router)
    }
}

pub struct RouteLayer {
    embedder: Arc<dyn Embedder>,
    index: Arc<dyn Index>,
    threshold: f64,
    llm: Arc<dyn LLM>,
}

impl RouteLayer {
    pub async fn add_routes(&mut self, routers: &mut [Router]) -> Result<(), RouteLayerError> {
        for router in routers.into_iter() {
            if router.embedding.is_none() {
                let embeddigns = self.embedder.embed_documents(&router.utterances).await?;
                router.embedding = Some(embeddigns);
            }
        }
        self.index.add(routers).await?;
        Ok(())
    }

    pub async fn delete_route<S: Into<String>>(
        &self,
        route_name: S,
    ) -> Result<(), RouteLayerError> {
        self.index.delete(&route_name.into()).await?;
        Ok(())
    }

    pub async fn get_routes(&self) -> Result<Vec<Router>, RouteLayerError> {
        let routes = self.index.get_routes().await?;
        Ok(routes)
    }

    pub async fn get_similar_routes<S: Into<String>>(
        &self,
        query: S,
        top_k: usize,
    ) -> Result<Vec<Router>, RouteLayerError> {
        todo!()
    }

    pub async fn route<S: Into<String>>(
        &self,
        query: S,
    ) -> Result<Option<Router>, RouteLayerError> {
        todo!()
    }

    pub async fn dynamic_route<S: Into<S>>(&self, _query: S) -> Result<String, RouteLayerError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_route_layer_builder() {
        let politics_route = Router::new(
            "politics",
            &[
                "isn't politics the best thing ever",
                "why don't you tell me about your political opinions",
                "don't you just love the president",
                "they're going to destroy this country!",
                "they will save the country!",
            ],
        );
        let chitchat_route = Router::new(
            "chitchat",
            &[
                "how's the weather today?",
                "how's the weather today?",
                "how are things going?",
                "lovely weather today",
                "the weather is horrendous",
                "let's go to the chippy",
            ],
        );
        let router_layer = RouteLayerBuilder::default()
            .embedder(OpenAiEmbedder::default())
            .add_route(politics_route)
            .add_route(chitchat_route)
            .threshold(0.5)
            .build()
            .await
            .unwrap();
        let routes = router_layer.route("Whats you favorite car").await.unwrap();
        println!("{:?}", routes.unwrap().name);
    }
}
