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

impl From<ResponseFormat> for async_openai::types::ResponseFormat {
    fn from(value: ResponseFormat) -> Self {
        match value {
            ResponseFormat::Text => async_openai::types::ResponseFormat::Text,
            ResponseFormat::JsonObject => async_openai::types::ResponseFormat::JsonObject,
            ResponseFormat::JsonSchema {
                name,
                description,
                schema,
                strict,
            } => async_openai::types::ResponseFormat::JsonSchema {
                json_schema: async_openai::types::ResponseFormatJsonSchema {
                    name,
                    description,
                    schema,
                    strict,
                },
            },
        }
    }
}
