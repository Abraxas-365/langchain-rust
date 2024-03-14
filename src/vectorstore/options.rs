use std::sync::Arc;

use serde_json::Value;

use crate::embedding::embedder_trait::Embedder;

/// The `VecStoreOptions` struct is responsible for determining options when
/// interacting with a Vector Store. The options include `name_space`, `score_threshold`,
/// `filters`, and `embedder`.
///
/// # Usage
/// ```rust,ignore
/// let options = VecStoreOptions::new()
///     .with_name_space("my_custom_namespace")
///     .with_score_threshold(0.5)
///     .with_filters(json!({"genre": "Sci-Fi"}))
///     .with_embedder(my_embedder);
/// ```
pub struct VecStoreOptions {
    pub name_space: Option<String>,
    pub score_threshold: Option<f32>,
    pub filters: Option<Value>,
    pub embedder: Option<Arc<dyn Embedder>>,
}

impl Default for VecStoreOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl VecStoreOptions {
    pub fn new() -> Self {
        VecStoreOptions {
            name_space: None,
            score_threshold: None,
            filters: None,
            embedder: None,
        }
    }

    pub fn with_name_space<S: Into<String>>(mut self, name_space: S) -> Self {
        self.name_space = Some(name_space.into());
        self
    }

    pub fn with_score_threshold(mut self, score_threshold: f32) -> Self {
        self.score_threshold = Some(score_threshold);
        self
    }

    pub fn with_filters(mut self, filters: Value) -> Self {
        self.filters = Some(filters);
        self
    }

    pub fn with_embedder<E: Embedder + 'static>(mut self, embedder: E) -> Self {
        self.embedder = Some(Arc::new(embedder));
        self
    }
}
