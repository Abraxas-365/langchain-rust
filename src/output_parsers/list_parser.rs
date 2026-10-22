use super::{
    typed::{string_output_parser, strip_reasoning},
    OutputParserError, TypedOutputParser,
};

/// Parses `foo, bar, baz` into `["foo", "bar", "baz"]`.
///
/// Items are trimmed and empty items dropped. As a chain output parser it yields a JSON
/// array (`["foo","bar","baz"]`), so items containing commas survive.
#[derive(Debug, Clone, Default)]
pub struct CommaSeparatedListParser;

impl CommaSeparatedListParser {
    pub fn new() -> Self {
        Self
    }
}

impl TypedOutputParser for CommaSeparatedListParser {
    type Output = Vec<String>;

    fn format_instructions(&self) -> String {
        "Your response should be a list of comma separated values, eg: `foo, bar, baz`".to_string()
    }

    fn parse_output(&self, text: &str) -> Result<Vec<String>, OutputParserError> {
        Ok(strip_reasoning(text)
            .split(',')
            .map(str::trim)
            .filter(|item| !item.is_empty())
            .map(str::to_string)
            .collect())
    }
}

string_output_parser!(CommaSeparatedListParser, |items| {
    serde_json::to_string(&items).expect("a list of strings serializes")
});
