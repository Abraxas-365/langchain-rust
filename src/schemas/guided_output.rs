use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum GuidedOutput {
    Choice { guided_choice: Vec<String> },
    Regex { guiude_regex: String, stop: String },
    Json { guided_json: serde_json::Value },
    Grammar { guided_grammar: String },
    WhitspacePattern { guided_whitespace: String },
}
