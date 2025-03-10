use core::fmt;
use std::collections::HashMap;
use std::string::String;
use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use serde_json::{json, Value};

#[async_trait]
pub trait Tool: Send + Sync {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "> {}: {}", self.name(), self.description())
    }

    /// Returns the name of the tool.
    fn name(&self) -> String;

    /// Provides a description of what the tool does and when to use it.
    fn description(&self) -> String;
    /// This are the parametters for OpenAi-like function call.
    /// You should return a jsnon like this one
    /// ```json
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

    /// Processes an input string and executes the tool's functionality, returning a `Result`.
    ///
    /// This function utilizes `parse_input` to parse the input and then calls `run`.
    /// Its used by the Agent
    async fn call(&self, input: &Value) -> Result<String, Box<dyn Error>> {
        let input = self.parse_input(input).await;
        self.run(input).await
    }

    /// Executes the core functionality of the tool.
    ///
    /// Example implementation:
    /// ```rust,ignore
    /// async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
    ///     let input_str = input.as_str().ok_or("Input should be a string")?;
    ///     self.simple_search(input_str).await
    /// }
    /// ```
    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>>;

    /// Parses the input string, which could be a JSON value or a raw string, depending on the LLM model.
    ///
    /// Implement this function to extract the parameters needed for your tool. If a simple
    /// string is sufficient, the default implementation can be used.
    async fn parse_input(&self, input: &Value) -> Value {
        input.clone()
    }
}

impl fmt::Display for dyn Tool {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "> {}: {}", self.name(), self.description())
    }
}

pub fn map_tools(tools: Vec<Arc<dyn Tool>>) -> HashMap<String, Arc<dyn Tool>> {
    tools
        .into_iter()
        .map(|tool| (tool.name().to_lowercase().replace(" ", "_"), tool))
        .collect()
}
