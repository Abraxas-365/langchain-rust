use crate::schemas::convert::{ OpenAIFromLangchain, TryOpenAiFromLangchain};
use crate::tools::Tool;
use async_openai::types::{ChatCompletionNamedToolChoice, ChatCompletionTool, ChatCompletionToolArgs, ChatCompletionToolChoiceOption, ChatCompletionToolType, FunctionName, FunctionObjectArgs};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::ops::Deref;

#[derive(Clone, Debug)]
pub enum FunctionCallBehavior {
    None,
    Auto,
    Named(String),
}

impl OpenAIFromLangchain<FunctionCallBehavior> for ChatCompletionToolChoiceOption {
    fn from_langchain(langchain: FunctionCallBehavior) -> Self {
        match langchain {
            FunctionCallBehavior::Auto => ChatCompletionToolChoiceOption::Auto,
            FunctionCallBehavior::None => ChatCompletionToolChoiceOption::None,
            FunctionCallBehavior::Named(name) => {
                ChatCompletionToolChoiceOption::Named(ChatCompletionNamedToolChoice {
                    r#type: ChatCompletionToolType::Function,
                    function: FunctionName {
                        name: name.to_owned(),
                    },
                })
            }
        }
    }
}

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
            parameters: tool.parameters(),
        }
    }
}

impl TryOpenAiFromLangchain<FunctionDefinition> for ChatCompletionTool {
    type Error = async_openai::error::OpenAIError;
    fn try_from_langchain(langchain: FunctionDefinition) -> Result<Self, Self::Error> {
        let tool = FunctionObjectArgs::default()
            .name(langchain.name)
            .description(langchain.description)
            .parameters(langchain.parameters)
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

impl FunctionCallResponse {
    pub fn from_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}
