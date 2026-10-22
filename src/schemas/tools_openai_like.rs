use crate::schemas::convert::{OpenAIFromLangchain, TryOpenAiFromLangchain};
use crate::tools::Tool;
use async_openai::types::chat::{
    ChatCompletionNamedToolChoice, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ChatCompletionTools, FunctionName, FunctionObjectArgs, ToolChoiceOptions,
};
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
            FunctionCallBehavior::Auto => {
                ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::Auto)
            }
            FunctionCallBehavior::None => {
                ChatCompletionToolChoiceOption::Mode(ToolChoiceOptions::None)
            }
            FunctionCallBehavior::Named(name) => {
                ChatCompletionToolChoiceOption::Function(ChatCompletionNamedToolChoice {
                    function: FunctionName { name },
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
        let function = FunctionObjectArgs::default()
            .name(langchain.name)
            .description(langchain.description)
            .parameters(langchain.parameters)
            .build()?;

        Ok(ChatCompletionTool { function })
    }
}

impl TryOpenAiFromLangchain<FunctionDefinition> for ChatCompletionTools {
    type Error = async_openai::error::OpenAIError;
    fn try_from_langchain(langchain: FunctionDefinition) -> Result<Self, Self::Error> {
        ChatCompletionTool::try_from_langchain(langchain).map(ChatCompletionTools::Function)
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

impl FunctionDetail {
    /// The arguments as a JSON object, for providers that take tool input as an object
    /// (Anthropic, Ollama, Bedrock). Non-object JSON or plain text is wrapped as
    /// `{"input": ...}`, and empty arguments become `{}`.
    pub fn arguments_object(&self) -> Value {
        super::ToolCall::new("", "", self.arguments.clone()).arguments_value()
    }
}

impl FunctionCallResponse {
    pub fn from_str(s: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;
    use std::{error::Error, sync::Arc};

    struct MyTool;

    #[async_trait]
    impl Tool for MyTool {
        fn name(&self) -> String {
            " Web Search ".into()
        }
        fn description(&self) -> String {
            "searches".into()
        }
        async fn run(&self, _input: Value) -> Result<String, Box<dyn Error>> {
            Ok(String::new())
        }
    }

    #[test]
    fn function_definition_normalizes_name() {
        let f = FunctionDefinition::new(" my cool tool ", "desc", json!({}));
        assert_eq!(f.name, "my_cool_tool");
    }

    #[test]
    fn function_definition_from_tool_uses_default_parameters() {
        let tool: Arc<dyn Tool> = Arc::new(MyTool);
        let f = FunctionDefinition::from_langchain_tool(&tool);
        assert_eq!(f.name, "Web_Search");
        assert_eq!(f.description, "searches");
        assert_eq!(f.parameters["required"], json!(["input"]));
        assert_eq!(
            f.parameters["properties"]["input"]["description"],
            "searches"
        );
    }

    #[test]
    fn function_call_response_parses() {
        let r = FunctionCallResponse::from_str(
            r#"{"id":"c1","type":"function","function":{"name":"calc","arguments":"{\"a\":1}"}}"#,
        )
        .unwrap();
        assert_eq!(r.id, "c1");
        assert_eq!(r.type_field, "function");
        assert_eq!(r.function.name, "calc");
        assert_eq!(r.function.arguments, "{\"a\":1}");
        assert!(FunctionCallResponse::from_str("{}").is_err());
    }
}
