use std::sync::Arc;

use futures_util::future::try_join_all;

use crate::embedding::Embedder;

use super::{RouteLayerBuilderError, RouteLayerError, Router};

pub struct RouteLayerBuilder {
    embedder: Option<Arc<dyn Embedder>>,
    routes: Vec<Router>,
    threshold: Option<f64>,
}
impl RouteLayerBuilder {
    pub fn new() -> Self {
        Self {
            embedder: None,
            routes: Vec::new(),
            threshold: None,
        }
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

    pub async fn build(self) -> Result<RouteLayer, RouteLayerBuilderError> {
        // Check if any routers lack an embedding and there's no global embedder provided.
        if self.embedder.is_none() {
            return Err(RouteLayerBuilderError::MissingEmbedderForRoutes);
        }

        Ok(RouteLayer {
            embedder: self.embedder.unwrap(), //it's safe to unwrap here because we checked for None above
            routes: self.routes,
            threshold: self.threshold.unwrap_or(0.7),
        })
    }
}

/// `RouteLayer` manages routing based on text embeddings.
///
/// Maintains a set of routes and an embedder for generating embedding representations
/// of textual data. Prior to using `RouteLayer` to route queries based on embeddings,
/// you must call `init` to compute and assign embeddings to all routes that do not
/// already have them. This ensures that each route has an embedding representation
/// based on its associated utterances, enabling effective routing based on
/// cosine similarity to query embeddings.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust,ignore
/// let mut route_layer = RouteLayer::new(Arc::new(your_embedder), 0.5);
/// let route1=RouterBuilder::new("MyRouter1".to_string())
///     .utterances(&["Hello", "World"])
///     .build()
///     .unwrap();
///
/// let route2=RouterBuilder::new("MyRouter2".to_string())
///     .embedding(&[1, 2, 3, 4, 5]);
///     .build()
///     .unwrap();
/// route_layer.add_route(rout1,route2);
/// route_layer.init().await.unwrap(); // Ensures routes have embeddings
///
/// // Now, you can use `route_layer.route(...)` to route queries.
/// ```
///
/// Note: `init` must be called to compute embeddings for routes that require them.
/// This step is crucial for the proper functioning of the `route` method.
pub struct RouteLayer {
    embedder: Arc<dyn Embedder>,
    routes: Vec<Router>,
    threshold: f64,
}

impl RouteLayer {
    pub async fn init(&mut self) -> Result<(), RouteLayerError> {
        let futures_and_indices: Vec<_> = self
            .routes
            .iter_mut()
            .enumerate()
            .filter_map(|(index, route)| {
                let embedder = Arc::clone(&self.embedder);
                route.utterances.as_ref().and_then(|utterances| {
                    if utterances.is_empty() || route.embedding.is_some() {
                        None
                    } else {
                        Some(async move {
                            let embeddings = embedder.embed_documents(utterances).await?;
                            let combined_embedding: Vec<f64> = combine_embeddings(&embeddings);
                            Ok::<_, RouteLayerError>((index, combined_embedding))
                        })
                    }
                })
            })
            .collect();

        let results = try_join_all(futures_and_indices).await?;
        for (index, embedding) in results {
            if let Some(route) = self.routes.get_mut(index) {
                route.embedding = Some(embedding);
            }
        }

        Ok(())
    }

    pub async fn route<S: Into<String>>(
        &self,
        query: S,
    ) -> Result<Option<&Router>, RouteLayerError> {
        let embedding = self.embedder.embed_query(&query.into()).await?;
        let route = self.route_embedding(&embedding);
        Ok(route)
    }

    pub fn route_embedding(&self, input_embedding: &[f64]) -> Option<&Router> {
        self.routes
            .iter()
            .filter_map(|route| {
                route.embedding.as_ref().and_then(|route_embedding| {
                    let similarity = cosine_similarity(input_embedding, route_embedding);
                    // Check if the similarity meets or exceeds the threshold
                    if similarity >= self.threshold {
                        Some((route, similarity))
                    } else {
                        None
                    }
                })
            })
            // Find the route with the highest similarity
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(route, _)| route) // Return the route, discarding the similarity score
    }
}

fn combine_embeddings(embeddings: &[Vec<f64>]) -> Vec<f64> {
    embeddings
        .iter()
        // Initialize a vector with zeros based on the length of the first embedding vector.
        // It's assumed all embeddings have the same dimensions.
        .fold(
            vec![0f64; embeddings[0].len()],
            |mut accumulator, embedding_vec| {
                for (i, &value) in embedding_vec.iter().enumerate() {
                    accumulator[i] += value;
                }
                accumulator
            },
        )
        // Calculate the mean for each element across all embeddings.
        .iter()
        .map(|&sum| sum / embeddings.len() as f64)
        .collect()
}

fn cosine_similarity(vec1: &[f64], vec2: &[f64]) -> f64 {
    let dot_product: f64 = vec1.iter().zip(vec2.iter()).map(|(a, b)| a * b).sum();
    let magnitude_vec1: f64 = vec1.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    let magnitude_vec2: f64 = vec2.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
    dot_product / (magnitude_vec1 * magnitude_vec2)
}

#[cfg(test)]
mod tests {
    use crate::embedding::openai::OpenAiEmbedder;

    use super::*;

    #[test]
    fn test_route_embedding() {
        let threshold = 0.5;
        let embedder = Arc::new(OpenAiEmbedder::default());
        let routes = vec![
            Router {
                name: "Route1".to_string(),
                utterances: None,
                embedding: Some(vec![0.1, 0.2, 0.3]),
            },
            Router {
                name: "Route2".to_string(),
                utterances: None,
                embedding: Some(vec![0.4, 0.5, 0.6]),
            },
        ];

        let route_layer = RouteLayer {
            embedder,
            routes,
            threshold,
        };

        // Input embedding
        let input_embedding = vec![0.3, 0.4, 0.5];

        // Call the method
        let selected_route = route_layer.route_embedding(&input_embedding);

        // Check the result
        assert!(selected_route.is_some());
        assert_eq!(selected_route.unwrap().name, "Route2");
    }
}
