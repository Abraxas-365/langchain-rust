use std::{collections::HashMap, sync::Arc};

use crate::{
    embedding::Embedder,
    language_models::llm::LLM,
    semantic_router::{Index, RouteLayerError, Router},
};

pub enum AggregationMethod {
    Mean,
    Max,
    Sum,
}
impl AggregationMethod {
    pub fn aggregate(&self, values: &[f64]) -> f64 {
        match self {
            AggregationMethod::Sum => values.iter().sum(),
            AggregationMethod::Mean => values.iter().sum::<f64>() / values.len() as f64,
            AggregationMethod::Max => *values
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(&0.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RouteChoise {
    pub route: String,
    pub similarity_score: f64,
}

pub struct RouteLayer {
    pub(crate) embedder: Arc<dyn Embedder>,
    pub(crate) index: Box<dyn Index>,
    pub(crate) threshold: f64,
    pub(crate) llm: Arc<dyn LLM>,
    pub(crate) top_k: usize,
    pub(crate) aggregation_method: AggregationMethod,
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

    async fn filter_similar_routes(
        &self,
        query_vector: &[f64],
    ) -> Result<Vec<(String, f64)>, RouteLayerError> {
        let similar_routes = self.index.query(query_vector, self.top_k).await?;

        Ok(similar_routes
            .into_iter()
            .filter(|(_, score)| *score >= self.threshold)
            .collect())
    }

    fn compute_total_scores(&self, similar_routes: &[(String, f64)]) -> HashMap<String, f64> {
        let mut scores_by_route: HashMap<String, Vec<f64>> = HashMap::new();

        for (route_name, score) in similar_routes {
            scores_by_route
                .entry(route_name.to_owned())
                .or_insert_with(Vec::new)
                .push(*score);
        }

        scores_by_route
            .into_iter()
            .map(|(route, scores)| {
                let aggregated_score = self.aggregation_method.aggregate(&scores);
                (route, aggregated_score)
            })
            .collect()
    }

    fn find_top_route_and_scores(
        &self,
        total_scores: HashMap<String, f64>,
        scores_by_route: &HashMap<String, Vec<f64>>,
    ) -> (Option<String>, Vec<f64>) {
        let top_route = total_scores
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(route, _)| route);

        let mut top_scores = top_route
            .as_ref()
            .and_then(|route| scores_by_route.get(route))
            .unwrap_or(&vec![])
            .clone();

        top_scores.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        (top_route, top_scores)
    }

    pub async fn call<S: Into<String>>(
        &self,
        query: S,
    ) -> Result<Option<RouteChoise>, RouteLayerError> {
        let query_vector = self.embedder.embed_query(&query.into()).await?;

        self.call_embedding(&query_vector).await
    }

    pub async fn call_embedding(
        &self,
        embedding: &[f64],
    ) -> Result<Option<RouteChoise>, RouteLayerError> {
        let similar_routes = self.filter_similar_routes(&embedding).await?;

        if similar_routes.is_empty() {
            return Ok(None);
        }

        // Correctly collect scores by route manually
        let mut scores_by_route: HashMap<String, Vec<f64>> = HashMap::new();
        for (route_name, score) in &similar_routes {
            scores_by_route
                .entry(route_name.clone())
                .or_default()
                .push(*score);
        }

        let total_scores = self.compute_total_scores(&similar_routes);

        let (top_route, top_scores) =
            self.find_top_route_and_scores(total_scores, &scores_by_route);

        Ok(top_route.map(|route| RouteChoise {
            route,
            similarity_score: top_scores.get(0).copied().unwrap_or(0.0),
        }))
    }
}

#[cfg(test)]
mod tests {

    use crate::{embedding::openai::OpenAiEmbedder, semantic_router::RouteLayerBuilder};

    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_route_layer_builder() {
        let captial_route = Router::new(
            "captial",
            &[
                "Capital of France is Paris.",
                "What is the captial of France?",
            ],
        );
        let weather_route = Router::new(
            "temperature",
            &[
                "What is the temperature?",
                "Is it raining?",
                "Is it cloudy?",
            ],
        );
        let router_layer = RouteLayerBuilder::default()
            .embedder(OpenAiEmbedder::default())
            .add_route(captial_route)
            .add_route(weather_route)
            .aggregation_method(AggregationMethod::Sum)
            .build()
            .await
            .unwrap();
        let routes = router_layer
            .call("What is the temperature in Peru?")
            .await
            .unwrap();

        assert_eq!(routes.unwrap().route, "temperature");
    }
}
