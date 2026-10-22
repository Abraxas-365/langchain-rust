use serde_json::{Map, Value};

use super::{
    typed::{extract_json, preview, string_output_parser},
    OutputParserError, TypedOutputParser,
};

/// One field the model has to return, for [`StructuredOutputParser`].
#[derive(Debug, Clone, PartialEq)]
pub struct ResponseSchema {
    pub name: String,
    pub description: String,
}

impl ResponseSchema {
    pub fn new<N: Into<String>, D: Into<String>>(name: N, description: D) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }
}

/// Asks the model for a JSON object with the given fields and checks they're all there.
///
/// ```rust,ignore
/// let parser = StructuredOutputParser::new([
///     ResponseSchema::new("answer", "answer to the user's question"),
///     ResponseSchema::new("source", "website used to answer"),
/// ]);
/// let fields = parser.parse_output(&generation)?;
/// ```
///
/// The JSON may come in a fenced block or bare. Field values are kept as JSON (strings,
/// numbers, lists...); extra fields are kept too. As a chain output parser it yields the
/// object as JSON.
#[derive(Debug, Clone)]
pub struct StructuredOutputParser {
    schemas: Vec<ResponseSchema>,
}

impl StructuredOutputParser {
    pub fn new<I: IntoIterator<Item = ResponseSchema>>(schemas: I) -> Self {
        Self {
            schemas: schemas.into_iter().collect(),
        }
    }
}

impl TypedOutputParser for StructuredOutputParser {
    type Output = Map<String, Value>;

    fn format_instructions(&self) -> String {
        let fields: String = self
            .schemas
            .iter()
            .map(|s| format!("\t\"{}\": string // {}\n", s.name, s.description))
            .collect();
        format!(
            "The output should be a markdown code snippet formatted in the following schema, \
             including the leading and trailing \"```json\" and \"```\":\n\n```json\n{{\n{fields}}}\n```"
        )
    }

    fn parse_output(&self, text: &str) -> Result<Self::Output, OutputParserError> {
        let Value::Object(object) = extract_json(text)? else {
            return Err(OutputParserError::ParsingError(format!(
                "expected a JSON object: {}",
                preview(text)
            )));
        };
        let missing: Vec<_> = self
            .schemas
            .iter()
            .filter(|s| !object.contains_key(&s.name))
            .map(|s| s.name.as_str())
            .collect();
        if !missing.is_empty() {
            return Err(OutputParserError::ParsingError(format!(
                "output is missing the fields {missing:?}: {}",
                preview(text)
            )));
        }
        Ok(object)
    }
}

string_output_parser!(StructuredOutputParser, |object| Value::Object(object)
    .to_string());
