use std::collections::HashMap;

use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Document {
    pub page_content: String,
    pub metadata: HashMap<String, Value>,
    pub score: f64,
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
