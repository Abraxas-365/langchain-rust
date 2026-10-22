use std::marker::PhantomData;

use serde::de::DeserializeOwned;
use serde_json::Value;

use super::{
    typed::{extract_json, preview},
    OutputParser, OutputParserError, TypedOutputParser,
};

/// Parses the model's JSON answer straight into your own type.
///
/// ```rust,ignore
/// #[derive(Deserialize)]
/// struct Movie { title: String, year: u16, genres: Vec<String> }
///
/// let parser = JsonParser::<Movie>::new().with_schema(json!({
///     "type": "object",
///     "properties": {
///         "title": {"type": "string"},
///         "year": {"type": "integer"},
///         "genres": {"type": "array", "items": {"type": "string"}}
///     },
///     "required": ["title", "year", "genres"]
/// }));
/// let prompt_args = prompt_args! { "format_instructions" => parser.format_instructions(), ... };
/// let movie: Movie = parser.parse_output(&generation)?;
/// ```
///
/// The JSON may come in a fenced block, bare, or surrounded by prose. Without a schema the
/// instructions only ask for JSON, so describe the shape in your prompt. As a chain output
/// parser it checks that the JSON deserializes into `T` and yields it as compact JSON.
pub struct JsonParser<T> {
    schema: Option<Value>,
    // `fn() -> T`: the parser is Send + Sync whatever `T` is; no `T` is stored.
    _output: PhantomData<fn() -> T>,
}

impl<T> Default for JsonParser<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Clone for JsonParser<T> {
    fn clone(&self) -> Self {
        Self {
            schema: self.schema.clone(),
            _output: PhantomData,
        }
    }
}

impl<T> std::fmt::Debug for JsonParser<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonParser")
            .field("schema", &self.schema)
            .finish()
    }
}

impl<T> JsonParser<T> {
    pub fn new() -> Self {
        Self {
            schema: None,
            _output: PhantomData,
        }
    }

    /// A JSON schema of `T`, put in the format instructions.
    pub fn with_schema(mut self, schema: Value) -> Self {
        self.schema = Some(schema);
        self
    }
}

impl<T: DeserializeOwned> JsonParser<T> {
    fn parse_value(&self, text: &str) -> Result<(Value, T), OutputParserError> {
        let value = extract_json(text)?;
        let parsed = serde_json::from_value(value.clone()).map_err(|e| {
            OutputParserError::ParsingError(format!(
                "JSON doesn't match the expected shape: {e}: {}",
                preview(text)
            ))
        })?;
        Ok((value, parsed))
    }
}

impl<T: DeserializeOwned> TypedOutputParser for JsonParser<T> {
    type Output = T;

    fn format_instructions(&self) -> String {
        match &self.schema {
            Some(schema) => format!(
                "Your output should be JSON that conforms to this JSON schema, in a ```json code block:\n```json\n{}\n```",
                serde_json::to_string_pretty(schema).expect("a JSON value serializes")
            ),
            None => "Your output should be a JSON value in a ```json code block.".to_string(),
        }
    }

    fn parse_output(&self, text: &str) -> Result<T, OutputParserError> {
        self.parse_value(text).map(|(_, parsed)| parsed)
    }
}

#[async_trait::async_trait]
impl<T: DeserializeOwned> OutputParser for JsonParser<T> {
    async fn parse(&self, output: &str) -> Result<String, OutputParserError> {
        self.parse_value(output).map(|(value, _)| value.to_string())
    }
}
