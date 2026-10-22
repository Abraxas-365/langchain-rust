use super::{
    typed::{preview, string_output_parser, strip_reasoning},
    OutputParserError, TypedOutputParser,
};

/// Parses a yes/no answer into a `bool`.
///
/// Matching ignores case, surrounding whitespace, quotes/backticks and a trailing period.
/// Defaults: `YES`/`TRUE` are true, `NO`/`FALSE` are false. As a chain output parser it
/// yields `"true"` or `"false"`.
#[derive(Debug, Clone)]
pub struct BooleanParser {
    true_values: Vec<String>,
    false_values: Vec<String>,
}

impl Default for BooleanParser {
    fn default() -> Self {
        Self::new()
    }
}

impl BooleanParser {
    pub fn new() -> Self {
        Self {
            true_values: vec!["YES".into(), "TRUE".into()],
            false_values: vec!["NO".into(), "FALSE".into()],
        }
    }

    /// Replaces the words read as true and false (compared case-insensitively).
    pub fn with_values<T, F>(mut self, true_values: T, false_values: F) -> Self
    where
        T: IntoIterator,
        T::Item: Into<String>,
        F: IntoIterator,
        F::Item: Into<String>,
    {
        self.true_values = true_values
            .into_iter()
            .map(|v| v.into().to_uppercase())
            .collect();
        self.false_values = false_values
            .into_iter()
            .map(|v| v.into().to_uppercase())
            .collect();
        self
    }
}

impl TypedOutputParser for BooleanParser {
    type Output = bool;

    fn format_instructions(&self) -> String {
        format!(
            "Answer with only `{}` or `{}`.",
            self.true_values
                .first()
                .map(String::as_str)
                .unwrap_or("YES"),
            self.false_values
                .first()
                .map(String::as_str)
                .unwrap_or("NO"),
        )
    }

    fn parse_output(&self, text: &str) -> Result<bool, OutputParserError> {
        let normalized = strip_reasoning(text)
            .trim()
            .trim_end_matches('.')
            .trim_matches(|c| matches!(c, '\'' | '"' | '`'))
            .trim()
            .to_uppercase();
        if self.true_values.contains(&normalized) {
            Ok(true)
        } else if self.false_values.contains(&normalized) {
            Ok(false)
        } else {
            Err(OutputParserError::ParsingError(format!(
                "expected one of {:?} or {:?}, got {}",
                self.true_values,
                self.false_values,
                preview(text)
            )))
        }
    }
}

string_output_parser!(BooleanParser, |value| value.to_string());
