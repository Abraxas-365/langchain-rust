use std::sync::OnceLock;

use regex::Regex;
use serde_json::Value;

use super::OutputParserError;

/// A parser that turns raw model output into a typed value, and tells the model what
/// format to answer in.
///
/// Put [`format_instructions`](Self::format_instructions) into the prompt, then call
/// [`parse_output`](Self::parse_output) on the generation. Every parser here also
/// implements [`OutputParser`](super::OutputParser), so it can go into
/// `LLMChainBuilder::output_parser`; the chain's generation is then the parsed value in a
/// normalized text form (see each parser).
///
/// Reasoning blocks (`<think>...</think>`, as emitted by DeepSeek R1 or Qwen 3) at the
/// start of the output are skipped.
pub trait TypedOutputParser: Send + Sync {
    type Output;

    /// Text to put in the prompt so the model answers in the expected format.
    fn format_instructions(&self) -> String;

    fn parse_output(&self, text: &str) -> Result<Self::Output, OutputParserError>;
}

/// Drops `<think>...</think>` blocks at the start of a reasoning model's output.
pub(crate) fn strip_reasoning(text: &str) -> &str {
    let mut rest = text.trim_start();
    while let Some(after_open) = rest.strip_prefix("<think>") {
        match after_open.find("</think>") {
            Some(end) => rest = after_open[end + "</think>".len()..].trim_start(),
            None => break,
        }
    }
    rest
}

fn fenced_block() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"```[A-Za-z0-9_-]*[ \t]*\r?\n?([\s\S]*?)```").unwrap())
}

/// Finds the JSON value in a model answer: a fenced code block (```json or plain ```), else
/// the whole text, else the span from the first `{`/`[` to the last matching `}`/`]`.
pub(crate) fn extract_json(text: &str) -> Result<Value, OutputParserError> {
    let text = strip_reasoning(text);
    let mut candidates: Vec<&str> = fenced_block()
        .captures_iter(text)
        .filter_map(|c| c.get(1).map(|m| m.as_str()))
        .collect();
    candidates.push(text);
    for (open, close) in [('{', '}'), ('[', ']')] {
        if let (Some(start), Some(end)) = (text.find(open), text.rfind(close)) {
            if start < end {
                candidates.push(&text[start..=end]);
            }
        }
    }

    let mut first_err = None;
    for candidate in candidates {
        match serde_json::from_str::<Value>(candidate.trim()) {
            Ok(value) => return Ok(value),
            Err(e) => {
                first_err.get_or_insert(e);
            }
        }
    }
    Err(OutputParserError::ParsingError(format!(
        "no JSON found in output ({}): {}",
        first_err.map(|e| e.to_string()).unwrap_or_default(),
        preview(text)
    )))
}

/// The start of `text`, for error messages.
pub(crate) fn preview(text: &str) -> String {
    const MAX: usize = 200;
    match text.char_indices().nth(MAX) {
        Some((cut, _)) => format!("{:?}...", &text[..cut]),
        None => format!("{text:?}"),
    }
}

/// Implements the string [`OutputParser`](super::OutputParser) for a typed parser by
/// rendering its output with `$render`.
macro_rules! string_output_parser {
    ($ty:ty, |$out:ident| $render:expr) => {
        #[async_trait::async_trait]
        impl $crate::output_parsers::OutputParser for $ty {
            async fn parse(
                &self,
                output: &str,
            ) -> Result<String, $crate::output_parsers::OutputParserError> {
                let $out = $crate::output_parsers::TypedOutputParser::parse_output(self, output)?;
                Ok($render)
            }
        }
    };
}
pub(crate) use string_output_parser;
