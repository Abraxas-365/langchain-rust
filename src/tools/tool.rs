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
                "input": {
                    "type": "string",
                    "description":self.description()
                }
            },
            "required": ["input"]
        })
    }
    async fn call(&self, input: &str) -> Result<String, Box<dyn Error>> {
        let input = self.parse_input(input).await;
        self.run(input).await
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>>;

    async fn parse_input(&self, input: &str) -> Value {
        match serde_json::from_str::<Value>(input) {
            Ok(input) => {
                if input["input"].is_string() {
                    Value::String(input["input"].as_str().unwrap().to_string())
                } else {
                    Value::String(input.to_string())
                }
            }
            Err(_) => Value::String(input.to_string()),
        }
    }
}
