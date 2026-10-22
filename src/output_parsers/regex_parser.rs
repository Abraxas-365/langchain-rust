use std::collections::HashMap;

use regex::Regex;

use super::{
    typed::{preview, string_output_parser, strip_reasoning},
    OutputParserError, TypedOutputParser,
};

/// Extracts fields from free-form output with a regex: every named capture group
/// (`(?P<name>...)`) becomes a key.
///
/// ```rust,ignore
/// let parser = RegexParser::new(r"Score: (?P<score>\d+)\s+Reason: (?P<reason>.+)")?;
/// let fields = parser.parse_output("Score: 8\nReason: clear and short")?;
/// assert_eq!(fields["score"], "8");
/// ```
///
/// Groups that didn't take part in the match are left out. As a chain output parser it
/// yields the fields as a JSON object.
#[derive(Debug, Clone)]
pub struct RegexParser {
    regex: Regex,
    instructions: Option<String>,
}

impl RegexParser {
    /// Fails if `pattern` isn't a valid regex or has no named groups.
    pub fn new(pattern: &str) -> Result<Self, OutputParserError> {
        let regex = Regex::new(pattern)?;
        if regex.capture_names().flatten().next().is_none() {
            return Err(OutputParserError::ParsingError(format!(
                "regex {pattern:?} has no named capture groups"
            )));
        }
        Ok(Self {
            regex,
            instructions: None,
        })
    }

    /// Format instructions to use instead of the generic ones, e.g. an example answer.
    pub fn with_format_instructions<S: Into<String>>(mut self, instructions: S) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

impl TypedOutputParser for RegexParser {
    type Output = HashMap<String, String>;

    fn format_instructions(&self) -> String {
        self.instructions.clone().unwrap_or_else(|| {
            let fields: Vec<_> = self.regex.capture_names().flatten().collect();
            format!(
                "Your answer must match this regular expression: {}\nIt provides these fields: {}",
                self.regex.as_str(),
                fields.join(", ")
            )
        })
    }

    fn parse_output(&self, text: &str) -> Result<Self::Output, OutputParserError> {
        let captures = self.regex.captures(strip_reasoning(text)).ok_or_else(|| {
            OutputParserError::ParsingError(format!(
                "output doesn't match {:?}: {}",
                self.regex.as_str(),
                preview(text)
            ))
        })?;
        Ok(self
            .regex
            .capture_names()
            .flatten()
            .filter_map(|name| {
                captures
                    .name(name)
                    .map(|m| (name.to_string(), m.as_str().to_string()))
            })
            .collect())
    }
}

string_output_parser!(RegexParser, |fields| {
    // Sorted, so the text is stable whatever the map's order.
    let fields: std::collections::BTreeMap<_, _> = fields.into_iter().collect();
    serde_json::to_string(&fields).expect("a map of strings serializes")
});
