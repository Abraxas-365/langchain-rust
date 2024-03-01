use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub page_content: String,
    pub metadata: HashMap<String, Value>,
    pub score: f64,
}

impl Document {
    pub fn new(page_content: String) -> Self {
        Document {
            page_content,
            metadata: HashMap::new(),
            score: 0.0,
        }
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_score(mut self, score: f64) -> Self {
        self.score = score;
        self
    }
}

impl Default for Document {
    fn default() -> Self {
        Document {
            page_content: "".to_string(),
            metadata: HashMap::new(),
            score: 0.0,
        }
    }
}
