use std::collections::{BTreeMap, HashMap};

use regex::Regex;
use serde_json::{Map, Value};

use super::{
    typed::{preview, string_output_parser, strip_reasoning},
    OutputParserError, TypedOutputParser,
};

/// A parser output made of named fields, which [`CombiningParser`] can merge.
pub trait IntoFields {
    fn into_fields(self) -> Result<Map<String, Value>, OutputParserError>;
}

impl IntoFields for Map<String, Value> {
    fn into_fields(self) -> Result<Map<String, Value>, OutputParserError> {
        Ok(self)
    }
}

impl IntoFields for HashMap<String, String> {
    fn into_fields(self) -> Result<Map<String, Value>, OutputParserError> {
        // Sorted, so the merged map doesn't depend on hash order.
        let sorted: BTreeMap<_, _> = self.into_iter().collect();
        Ok(sorted
            .into_iter()
            .map(|(k, v)| (k, Value::String(v)))
            .collect())
    }
}

/// A JSON object (e.g. from `JsonParser<Value>`); anything else is an error.
impl IntoFields for Value {
    fn into_fields(self) -> Result<Map<String, Value>, OutputParserError> {
        match self {
            Value::Object(map) => Ok(map),
            other => Err(OutputParserError::ParsingError(format!(
                "expected a JSON object, got {other}"
            ))),
        }
    }
}

trait FieldsParser: Send + Sync {
    fn instructions(&self) -> String;
    fn parse_fields(&self, text: &str) -> Result<Map<String, Value>, OutputParserError>;
}

impl<P> FieldsParser for P
where
    P: TypedOutputParser,
    P::Output: IntoFields,
{
    fn instructions(&self) -> String {
        self.format_instructions()
    }
    fn parse_fields(&self, text: &str) -> Result<Map<String, Value>, OutputParserError> {
        self.parse_output(text)?.into_fields()
    }
}

/// Runs several parsers over one answer that has one part per parser, separated by blank
/// lines, and merges their fields.
///
/// ```rust
/// use langchain_rust::output_parsers::{
///     CombiningParser, RegexParser, ResponseSchema, StructuredOutputParser, TypedOutputParser,
/// };
///
/// let parser = CombiningParser::new(
///     StructuredOutputParser::new([
///         ResponseSchema::new("answer", "the answer"),
///         ResponseSchema::new("source", "a link to the source"),
///     ]),
///     RegexParser::new(r"Confidence: (?P<confidence>A|B|C)")?,
/// );
/// let fields = parser.parse_output(
///     "```json\n{\"answer\": \"Paris\", \"source\": \"wiki\"}\n```\n\nConfidence: A",
/// )?;
/// assert_eq!(fields["answer"], "Paris");
/// assert_eq!(fields["confidence"], "A");
/// # Ok::<(), langchain_rust::output_parsers::OutputParserError>(())
/// ```
///
/// Parts are matched to parsers in order, and their number must equal the number of
/// parsers (empty parts, e.g. trailing blank lines, don't count). So a part can't itself
/// contain a blank line. When two parsers return the same key, the later one wins. Any
/// parser whose output is a set of fields works: [`RegexParser`](super::RegexParser),
/// [`RegexDictParser`](super::RegexDictParser),
/// [`StructuredOutputParser`](super::StructuredOutputParser), `JsonParser<Value>`, or
/// your own (see [`IntoFields`]). As a chain output parser it yields the merged fields as
/// a JSON object.
///
/// Like LangChain's `CombiningOutputParser` and langchaingo's `Combining`.
pub struct CombiningParser {
    parsers: Vec<Box<dyn FieldsParser>>,
}

impl CombiningParser {
    /// Combines two parsers; add more with [`with`](Self::with).
    pub fn new<A, B>(first: A, second: B) -> Self
    where
        A: TypedOutputParser + 'static,
        A::Output: IntoFields,
        B: TypedOutputParser + 'static,
        B::Output: IntoFields,
    {
        Self {
            parsers: vec![Box::new(first), Box::new(second)],
        }
    }

    /// Adds a parser for the next part of the answer.
    pub fn with<P>(mut self, parser: P) -> Self
    where
        P: TypedOutputParser + 'static,
        P::Output: IntoFields,
    {
        self.parsers.push(Box::new(parser));
        self
    }
}

fn blank_line() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\r?\n[ \t]*\r?\n").expect("valid regex"))
}

impl TypedOutputParser for CombiningParser {
    type Output = Map<String, Value>;

    fn format_instructions(&self) -> String {
        let parts: Vec<String> = self
            .parsers
            .iter()
            .enumerate()
            .map(|(i, p)| format!("Part {}:\n{}", i + 1, p.instructions()))
            .collect();
        format!(
            "Your answer must have {} parts, in this order, separated by a blank line. No part may contain a blank line itself.\n\n{}",
            self.parsers.len(),
            parts.join("\n\n")
        )
    }

    fn parse_output(&self, text: &str) -> Result<Self::Output, OutputParserError> {
        let text = strip_reasoning(text);
        let parts: Vec<&str> = blank_line()
            .split(text)
            .map(str::trim)
            .filter(|part| !part.is_empty())
            .collect();
        if parts.len() != self.parsers.len() {
            return Err(OutputParserError::ParsingError(format!(
                "expected {} parts separated by blank lines, got {}: {}",
                self.parsers.len(),
                parts.len(),
                preview(text)
            )));
        }
        let mut out = Map::new();
        for (i, (parser, part)) in self.parsers.iter().zip(parts).enumerate() {
            let fields = parser.parse_fields(part).map_err(|e| match e {
                OutputParserError::ParsingError(msg) => {
                    OutputParserError::ParsingError(format!("part {}: {msg}", i + 1))
                }
                other => other,
            })?;
            out.extend(fields);
        }
        Ok(out)
    }
}

string_output_parser!(CombiningParser, |fields| {
    let fields: BTreeMap<_, _> = fields.into_iter().collect();
    serde_json::to_string(&fields).expect("JSON values serialize")
});
