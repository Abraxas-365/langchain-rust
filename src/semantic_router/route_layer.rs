use std::{collections::HashMap, sync::Arc};

use crate::{
    embedding::{openai::OpenAiEmbedder, Embedder},
    language_models::llm::LLM,
    llm::openai::OpenAI,
    semantic_router::utils::cosine_similarity,
};

use super::{Index, LocalIndex, RouteLayerBuilderError, RouteLayerError, Router};

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
    agregration_method: AgregationMethod,
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
            top_k: 5,
            agregration_method: AgregationMethod::Sum,
        }
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
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

    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }

    pub fn add_route(mut self, route: Router) -> Self {
        self.routes.push(route);
        self
    }

    pub fn agregation_method(mut self, agegration_method: AgregationMethod) -> Self {
        self.agregration_method = agegration_method;
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
            threshold: self.threshold.unwrap_or(0.7),
            top_k: self.top_k,
            agegration_method: self.agregration_method,
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

pub enum AgregationMethod {
    Mean,
    Max,
    Sum,
}
impl AgregationMethod {
    pub fn aggregate(&self, values: &[f64]) -> f64 {
        match self {
            AgregationMethod::Sum => values.iter().sum(),
            AgregationMethod::Mean => values.iter().sum::<f64>() / values.len() as f64,
            AgregationMethod::Max => *values
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0),
        }
    }
}

pub struct RouteLayer {
    embedder: Arc<dyn Embedder>,
    index: Box<dyn Index>,
    threshold: f64,
    llm: Arc<dyn LLM>,
    top_k: usize,
    agegration_method: AgregationMethod,
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
        &mut self,
        route_name: S,
    ) -> Result<(), RouteLayerError> {
        self.index.delete(&route_name.into()).await?;
        Ok(())
    }

    pub async fn get_routes(&self) -> Result<Vec<Router>, RouteLayerError> {
        let routes = self.index.get_routes().await?;
        Ok(routes)
    }
    pub async fn call<S: Into<String>>(
        &self,
        query: S,
    ) -> Result<(Option<String>, Vec<f64>), RouteLayerError> {
        let query_vector = self.embedder.embed_query(&query.into()).await?;

        // Getting top_k similar routes and their scores
        let similar_routes = self.index.query(&query_vector, self.top_k).await?;
        for (route, score) in &similar_routes {
            println!("Route: {}, Score: {}", route, score);
        }

        if similar_routes.is_empty() {
            return Ok((None, vec![]));
        }

        // Aggregate scores by route
        let mut scores_by_route: HashMap<String, Vec<f64>> = HashMap::new();
        for (route_name, score) in similar_routes {
            scores_by_route
                .entry(route_name)
                .or_insert_with(Vec::new)
                .push(score);
        }

        // Calculate total score for each route using the selected aggregation method
        let mut total_scores: HashMap<String, f64> = HashMap::new();
        for (route, scores) in &scores_by_route {
            let aggregated_score = self.agegration_method.aggregate(scores);
            total_scores.insert(route.to_string(), aggregated_score);
        }

        // Finding the route with the highest aggregated score
        let top_route = total_scores
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(route, _)| route);

        // Getting associated scores for the top_route
        let top_scores = top_route
            .as_ref()
            .and_then(|route| scores_by_route.get(route))
            .unwrap_or(&vec![])
            .clone();

        Ok((top_route, top_scores))
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
        let politics_route = Router::new("politics", &["arm"]);
        let chitchat_route = Router::new("chitchat", &["car"]);
        let router_layer = RouteLayerBuilder::default()
            .embedder(OpenAiEmbedder::default())
            .add_route(politics_route)
            .add_route(chitchat_route)
            .agregation_method(AgregationMethod::Sum)
            .build()
            .await
            .unwrap();
        let routes = router_layer
            .call("Let’s Take a Selfie-Ms. Idea Robber.")
            .await
            .unwrap();
        println!("{:?}", routes);
    }
}
