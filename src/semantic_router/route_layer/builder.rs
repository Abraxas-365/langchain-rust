use std::sync::Arc;

use crate::{
    embedding::{openai::OpenAiEmbedder, Embedder},
    language_models::llm::LLM,
    llm::openai::OpenAI,
    semantic_router::{Index, MemoryIndex, RouteLayerBuilderError, Router},
};

use super::{AggregationMethod, RouteLayer};

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
    index: Option<Box<dyn Index>>,
    llm: Option<Arc<dyn LLM>>,
    top_k: usize,
    aggregation_method: AggregationMethod,
}
impl Default for RouteLayerBuilder {
    fn default() -> Self {
        Self::new()
            .embedder(OpenAiEmbedder::default())
            .llm(OpenAI::default())
            .index(MemoryIndex::new())
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
            top_k: 5,
            aggregation_method: AggregationMethod::Sum,
        }
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        let mut top_k = top_k;
        if top_k == 0 {
            log::warn!("top_k cannot be 0, setting it to 1");
            top_k = 1;
        }
        self.top_k = top_k;
        self
    }

    pub fn llm<L: LLM + 'static>(mut self, llm: L) -> Self {
        self.llm = Some(Arc::new(llm));
        self
    }

    pub fn index<I: Index + 'static>(mut self, index: I) -> Self {
        self.index = Some(Box::new(index));
        self
    }

    pub fn embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }

    /// The threshold is the minimum similarity score that a route must have to be considered.
    /// This depends on the similarity metric used by the embedder.
    /// For open ai text-embedding-ada-002, the best threshold is 0.82
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn add_route(mut self, route: Router) -> Self {
        self.routes.push(route);
        self
    }

    pub fn aggregation_method(mut self, aggregation_method: AggregationMethod) -> Self {
        self.aggregation_method = aggregation_method;
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

        let mut router = RouteLayer {
            embedder: self.embedder.unwrap(), //it's safe to unwrap here because we checked for None above
            index: self.index.unwrap(),
            llm: self.llm.unwrap(),
            threshold: self.threshold.unwrap_or(0.82),
            top_k: self.top_k,
            aggregation_method: self.aggregation_method,
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
