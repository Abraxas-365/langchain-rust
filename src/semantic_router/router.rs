use std::hash::{Hash, Hasher};

#[derive(Debug, Clone)]
pub struct Router {
    pub name: String,
    pub utterances: Vec<String>,
    pub embedding: Option<Vec<Vec<f64>>>,
    pub similarity: Option<f64>,
    pub tool_description: Option<String>,
}
impl Router {
    pub fn new<S: AsRef<str>>(name: &str, utterances: &[S]) -> Self {
        Self {
            name: name.into(),
            utterances: utterances.iter().map(|s| s.as_ref().to_string()).collect(),
            embedding: None,
            similarity: None,
            tool_description: None,
        }
    }

    pub fn with_embedding(mut self, embedding: Vec<Vec<f64>>) -> Self {
        self.embedding = Some(embedding);
        self
    }

    pub fn with_tool_description<S: Into<String>>(mut self, tool_description: S) -> Self {
        self.tool_description = Some(tool_description.into());
        self
    }

    pub fn with_similarity(mut self, similarity: f64) -> Self {
        self.similarity = Some(similarity);
        self
    }
}

impl Eq for Router {}
impl PartialEq for Router {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Hash for Router {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}
