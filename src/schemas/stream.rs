use serde_json::Value;

pub struct StreamData {
    pub value: Value,
    pub content: String,
}
impl StreamData {
    pub fn new<S: Into<String>>(value: Value, content: S) -> Self {
        Self {
            value,
            content: content.into(),
        }
    }
}
