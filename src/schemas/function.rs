use crate::tools::tool_field::ToolField;
use crate::tools::Tool;
use async_openai::types::{
    ChatCompletionTool, ChatCompletionToolArgs, ChatCompletionToolType, FunctionObjectArgs,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ops::Deref;
use std::str::FromStr;

#[derive(Clone, Debug)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl FunctionDefinition {
    pub fn new(name: &str, description: &str, parameters: Value) -> Self {
        FunctionDefinition {
            name: name.trim().replace(" ", "_"),
            description: description.to_string(),
            parameters,
        }
    }

    /// Generic function that can be used with both Arc<Tool>, Box<Tool>, and direct references
    pub fn from_langchain_tool<T>(tool: &T) -> FunctionDefinition
    where
        T: Deref<Target = dyn Tool> + ?Sized,
    {
        FunctionDefinition {
            name: tool.name().trim().replace(" ", "_"),
            description: tool.description(),
            parameters: tool.parameters().to_openai_field(),
        }
    }
}

impl TryFrom<FunctionDefinition> for ChatCompletionTool {
    type Error = async_openai::error::OpenAIError;

    fn try_from(value: FunctionDefinition) -> Result<Self, Self::Error> {
        let tool = FunctionObjectArgs::default()
            .name(value.name)
            .description(value.description)
            .parameters(value.parameters)
            .build()?;

        ChatCompletionToolArgs::default()
            .r#type(ChatCompletionToolType::Function)
            .function(tool)
            .build()
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FunctionCallResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub type_field: String,
    pub function: FunctionDetail,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FunctionDetail {
    pub name: String,
    ///this should be an string, and this should be passed to the tool, to
    ///then be deserilised inside the tool, becuase just the tools knows the names of the arguments.
    pub arguments: String,
}

impl FromStr for FunctionCallResponse {
    type Err = serde_json::Error;

    fn from_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}
