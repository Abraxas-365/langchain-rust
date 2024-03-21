use std::error::Error;
use std::string::String;

use async_trait::async_trait;
use serde_json::{json, Value};

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> String;
    fn description(&self) -> String;
    ///This are the parametters for OpenAi like function call.
    ///You should return a jsnon like this one
    ///```json
    /// {
    ///     "type": "object",
    ///     "properties": {
    ///         "command": {
    ///             "type": "string",
    ///             "description": "The raw command you want executed"
    ///                 }
    ///     },
    ///     "required": ["command"]
    /// }
    ///
    /// If there s no implementation the defaul will be the self.description()
    ///```
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
                "properties": {
                "command": {
                    "type": "string",
                    "description":self.description()
                }
            },
            "required": ["command"]
        })
    }
    async fn call(&self, input: &str) -> Result<String, Box<dyn Error>>;
}
