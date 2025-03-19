use std::sync::Arc;

use futures_util::future::try_join_all;

use crate::{
    chain::{LLMChain, LLMChainBuilder},
    embedding::{openai::OpenAiEmbedder, Embedder},
    language_models::llm::LLM,
    llm::openai::OpenAI,
    schemas::MessageType,
    semantic_router::{Index, MemoryIndex, RouteLayerBuilderError, Router},
    template::MessageTemplate,
};

use super::{AggregationMethod, RouteLayer};

/// A builder for creating a `RouteLayer`.
///```rust,ignore
/// let captial_route = Router::new(
///     "captial",
///     &[
///         "Capital of France is Paris.",
///         "What is the captial of France?",
///     ],
/// );
/// let weather_route = Router::new(
///     "temperature",
///     &[
///         "What is the temperature?",
///         "Is it raining?",
///         "Is it cloudy?",
///     ],
/// );
/// let router_layer = RouteLayerBuilder::default()
///     .embedder(OpenAiEmbedder::default())
///     .add_route(captial_route)
///     .add_route(weather_route)
///     .aggregation_method(AggregationMethod::Sum)
///     .threshold(0.82)
///     .build()
///     .await
///     .unwrap();
/// ```
pub struct RouteLayerBuilder {
    embedder: Option<Arc<dyn Embedder>>,
    routes: Vec<Router>,
    threshold: Option<f64>,
    index: Option<Box<dyn Index>>,
    llm: Option<LLMChain>,
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
        let prompt = MessageTemplate::from_jinja2(
            MessageType::HumanMessage,
            r#"
            You should Generate the input for the following tool.
            Tool description:{{description}}.
            Input query context to generate the input for the tool :{{query}}

            Tool Input:
            "#,
        );
        let chain = LLMChainBuilder::new()
            .prompt(prompt)
            .llm(llm)
            .build()
            .unwrap(); //safe to unwrap
        self.llm = Some(chain);
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

        let embedding_futures = self
            .routes
            .iter_mut()
            .filter_map(|route| {
                if route.embedding.is_none() {
                    Some(router.embedder.embed_documents(&route.utterances))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let embeddings = try_join_all(embedding_futures).await?;

        for (route, embedding) in self
            .routes
            .iter_mut()
            .filter(|r| r.embedding.is_none())
            .zip(embeddings)
        {
            route.embedding = Some(embedding);
        }

        // Add routes to the index.
        router.index.add(&self.routes).await?;

        Ok(router)
    }
}
