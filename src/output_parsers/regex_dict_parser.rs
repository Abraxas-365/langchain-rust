use std::collections::{BTreeMap, HashMap};

use regex::Regex;

use super::{
    typed::{preview, string_output_parser, strip_reasoning},
    OutputParserError, TypedOutputParser,
};

/// Reads labelled values (`Label: value`) out of free-form output, one per key.
///
/// ```rust
/// use langchain_rust::output_parsers::{RegexDictParser, TypedOutputParser};
///
/// let parser = RegexDictParser::new([("action", "Action"), ("action_input", "Action Input")])
///     .with_no_update_value("N/A");
/// let fields = parser.parse_output("Action: search\nAction Input: rust 2024")?;
/// assert_eq!(fields["action_input"], "rust 2024");
/// # Ok::<(), langchain_rust::output_parsers::OutputParserError>(())
/// ```
///
/// A value runs from the label to the end of its sentence: the first `.` followed by a
/// space (the `.` is kept), else the end of the line. So `3.5` or a URL stay whole. Labels
/// are matched literally. Every label must appear exactly once: a missing or repeated label
/// is an error. A value equal to the [no-update value](Self::with_no_update_value) (e.g.
/// `N/A`) leaves its key out. As a chain output parser it yields the fields as a JSON
/// object.
///
/// Like LangChain's `RegexDictParser` and langchaingo's `RegexDict`.
#[derive(Debug, Clone)]
pub struct RegexDictParser {
    /// Output key, label, and the regex that finds the label's value.
    fields: Vec<(String, String, Regex)>,
    no_update_value: Option<String>,
}

impl RegexDictParser {
    /// `output_key_to_label` maps each output key to the label that precedes its value in
    /// the text. The keys keep the given order in the format instructions.
    pub fn new<I, K, L>(output_key_to_label: I) -> Self
    where
        I: IntoIterator<Item = (K, L)>,
        K: Into<String>,
        L: Into<String>,
    {
        let fields = output_key_to_label
            .into_iter()
            .map(|(key, label)| {
                let label = label.into();
                let regex = Regex::new(&format!(
                    r"(?:{}):[ \t]?(?P<value>[^\n]*)",
                    regex::escape(&label)
                ))
                .expect("an escaped label makes a valid regex");
                (key.into(), label, regex)
            })
            .collect();
        Self {
            fields,
            no_update_value: None,
        }
    }

    /// A value meaning "unknown" (e.g. `N/A`): keys with it are left out of the result.
    pub fn with_no_update_value<S: Into<String>>(mut self, value: S) -> Self {
        self.no_update_value = Some(value.into());
        self
    }
}

impl TypedOutputParser for RegexDictParser {
    type Output = HashMap<String, String>;

    fn format_instructions(&self) -> String {
        let lines: Vec<String> = self
            .fields
            .iter()
            .map(|(_, label, _)| format!("{label}: <value>"))
            .collect();
        let mut text = format!(
            "Your answer must contain these lines, each once:\n{}",
            lines.join("\n")
        );
        if let Some(none) = &self.no_update_value {
            text.push_str(&format!("\nWrite {none} as the value you don't know."));
        }
        text
    }

    fn parse_output(&self, text: &str) -> Result<Self::Output, OutputParserError> {
        let text = strip_reasoning(text);
        let mut out = HashMap::with_capacity(self.fields.len());
        for (key, label, regex) in &self.fields {
            let mut matches = regex.captures_iter(text);
            let value = match (matches.next(), matches.next()) {
                (Some(m), None) => first_sentence(&m["value"]).to_string(),
                (None, _) => {
                    return Err(OutputParserError::ParsingError(format!(
                        "no `{label}:` in output: {}",
                        preview(text)
                    )))
                }
                (Some(_), Some(_)) => {
                    return Err(OutputParserError::ParsingError(format!(
                        "`{label}:` appears more than once in output: {}",
                        preview(text)
                    )))
                }
            };
            if self.no_update_value.as_deref() == Some(value.as_str()) {
                continue;
            }
            out.insert(key.clone(), value);
        }
        Ok(out)
    }
}

string_output_parser!(RegexDictParser, |fields| {
    let fields: BTreeMap<_, _> = fields.into_iter().collect();
    serde_json::to_string(&fields).expect("a map of strings serializes")
});

/// `line` up to and including the first `.` that ends a sentence (followed by whitespace),
/// trimmed. A `.` inside a number or URL doesn't end it.
fn first_sentence(line: &str) -> &str {
    let line = line.trim();
    let end = line
        .match_indices('.')
        .find(|(dot, _)| line[dot + 1..].starts_with(char::is_whitespace));
    match end {
        Some((dot, _)) => &line[..=dot],
        None => line,
    }
}
