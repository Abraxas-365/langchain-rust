use crate::schemas::convert::OpenAIFromLangchain;
use async_openai::types::chat::{
    ResponseFormat as OpenAIResponseFormat, ResponseFormatJsonSchema as OpenAIJsonSchema,
};

#[derive(Clone, Debug)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        description: Option<String>,
        name: String,
        schema: Option<serde_json::Value>,
        strict: Option<bool>,
    },
}

impl OpenAIFromLangchain<ResponseFormat> for OpenAIResponseFormat {
    fn from_langchain(langchain: ResponseFormat) -> Self {
        match langchain {
            ResponseFormat::Text => OpenAIResponseFormat::Text,
            ResponseFormat::JsonObject => OpenAIResponseFormat::JsonObject,
            ResponseFormat::JsonSchema {
                name,
                description,
                schema,
                strict,
            } => OpenAIResponseFormat::JsonSchema {
                json_schema: OpenAIJsonSchema {
                    name,
                    description,
                    // The OpenAI API requires a schema; an empty object accepts any JSON.
                    schema: schema.unwrap_or_else(|| serde_json::json!({})),
                    strict,
                },
            },
        }
    }
}
