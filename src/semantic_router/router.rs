use super::RouterBuilderError;

/// A builder for creating a `Router` instance.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust,ignore
/// # use your_crate::{RouterBuilder, Router, RouterBuilderError};
/// // Initializing the builder with a name
/// let mut builder = RouterBuilder::new("MyRouter".to_string());
///
/// // Optionally adding utterances
/// builder = builder.utterances(&["Hello", "World"]);
///
/// // Optionally adding an embedding
/// builder = builder.embedding(&[1, 2, 3, 4, 5]);
///
/// // Attempt to build the Router, handling potential errors
/// let router = match builder.build() {
///     Ok(router) => router,
///     Err(RouterBuilderError::InvalidConfiguration) => {
///         // Handle the error, e.g., by logging or fixing the configuration
///         panic!("Invalid router configuration: either utterances or embedding must be provided, and utterances cannot be an empty vector.");
///     }
/// };
/// ```
pub struct RouterBuilder {
    pub name: String,
    pub utterances: Option<Vec<String>>,
    pub embedding: Option<Vec<f64>>,
}
impl RouterBuilder {
    pub fn new(name: String) -> Self {
        RouterBuilder {
            name,
            utterances: None,
            embedding: None,
        }
    }

    /// Add a list of utterances to the `RouterBuilder`.
    pub fn utterances<S: AsRef<str>>(mut self, utterances: &[S]) -> Self {
        self.utterances = Some(utterances.iter().map(|s| s.as_ref().to_owned()).collect());
        self
    }

    /// Add an embedding to the `RouterBuilder`.
    pub fn embedding(mut self, embedding: &[f64]) -> Self {
        self.embedding = Some(embedding.to_vec());
        self
    }

    pub fn build(self) -> Result<Router, RouterBuilderError> {
        if self.utterances.is_none() && self.embedding.is_none() {
            Err(RouterBuilderError::InvalidConfiguration)
        } else if self.utterances.as_ref().map_or(false, |v| v.is_empty()) {
            Err(RouterBuilderError::InvalidConfiguration)
        } else {
            Ok(Router {
                name: self.name,
                utterances: self.utterances,
                embedding: self.embedding,
            })
        }
    }
}

#[derive(Debug, Clone)]
pub struct Router {
    pub name: String,
    pub utterances: Option<Vec<String>>,
    pub embedding: Option<Vec<f64>>,
}
