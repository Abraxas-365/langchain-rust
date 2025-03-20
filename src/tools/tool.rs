use std::collections::HashMap;
use std::string::String;
use std::sync::Arc;
use std::{error::Error, fmt::Display};

use async_trait::async_trait;
use derive_new::new;
use serde_json::{json, Value};

#[async_trait]
pub trait Tool: Send + Sync {
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
    async fn call(&self, input: Value) -> Result<String, Box<dyn Error + Send + Sync>>;

    fn usage_limit(&self) -> Option<usize> {
        None
    }
}

#[async_trait]
pub trait ToolFunction: Default + Send + Sync + Into<Arc<dyn Tool>> {
    type Input: Send + Sync;
    type Result: Display + Send + Sync;

    fn name(&self) -> String;

    fn description(&self) -> String;

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

    /// Executes the core functionality of the tool.
    ///
    /// Example implementation:
    /// ```rust,ignore
    /// async fn run(&self, input: ToolInput) -> Result<String, Box<dyn Error>> {
    ///     self.simple_search(input).await
    /// }
    /// ```
    async fn run(&self, input: Self::Input) -> Result<Self::Result, Box<dyn Error + Send + Sync>>;

    /// Parses the input string, which could be a JSON value or a raw string, depending on the LLM model.
    ///
    /// Implement this function to extract the parameters needed for your tool. If a simple
    /// string is sufficient, the default implementation can be used.
    async fn parse_input(&self, input: Value) -> Result<Self::Input, Box<dyn Error + Send + Sync>>;

    fn usage_limit(&self) -> Option<usize> {
        None
    }
}

#[derive(new)]
#[repr(transparent)]
pub struct ToolWrapper<T>
where
    T: ToolFunction,
{
    tool: T,
}

impl Display for dyn Tool {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "> {}: {}", self.name(), self.description())
    }
}

impl<T> Display for ToolWrapper<T>
where
    T: ToolFunction,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "> {}: {}", self.name(), self.description())
    }
}

#[async_trait]
impl<T> Tool for ToolWrapper<T>
where
    T: ToolFunction,
{
    fn name(&self) -> String {
        self.tool.name()
    }

    fn description(&self) -> String {
        self.tool.description()
    }

    fn parameters(&self) -> Value {
        self.tool.parameters()
    }

    async fn call(&self, input: Value) -> Result<String, Box<dyn Error + Send + Sync>> {
        let input = self.tool.parse_input(input).await?;
        let result = self.tool.run(input).await?;

        Ok(result.to_string())
    }

    fn usage_limit(&self) -> Option<usize> {
        self.tool.usage_limit()
    }
}

pub fn map_tools(tools: Vec<Arc<dyn Tool>>) -> HashMap<String, Arc<dyn Tool>> {
    tools
        .into_iter()
        .map(|tool| (tool.name().to_lowercase().replace(" ", "_"), tool))
        .collect()
}
