use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{json, Value};

use super::{FunctionCallResponse, FunctionDetail};

/// A tool/function call requested by a model, the same shape for every provider
/// (OpenAI, Claude, Ollama, Bedrock, ...).
///
/// `arguments` is the raw JSON string produced by the model; only the tool knows how to
/// interpret it. Serializes to the OpenAI wire format
/// (`{"id","type":"function","function":{"name","arguments"}}`), which is also what
/// [`Message::tool_calls`](super::Message) holds.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// Builds a call from structured arguments (what Claude, Ollama and Bedrock return).
    pub fn from_value(id: impl Into<String>, name: impl Into<String>, arguments: &Value) -> Self {
        Self::new(id, name, arguments.to_string())
    }

    /// The arguments as a JSON object, for providers that take tool input as an object
    /// (Anthropic, Ollama, Bedrock). Non-object JSON or plain text is wrapped as
    /// `{"input": ...}`, and empty arguments become `{}`.
    pub fn arguments_value(&self) -> Value {
        if self.arguments.trim().is_empty() {
            return Value::Object(Default::default());
        }
        match serde_json::from_str::<Value>(&self.arguments) {
            Ok(v @ Value::Object(_)) => v,
            Ok(other) => json!({ "input": other }),
            Err(_) => json!({ "input": self.arguments }),
        }
    }

    /// Parses an OpenAI-style JSON array of tool calls (e.g. a `generation` string from an
    /// LLM that encodes tool calls as text). Returns `None` if it is not one.
    pub fn parse_list(s: &str) -> Option<Vec<ToolCall>> {
        let trimmed = s.trim_start();
        if !trimmed.starts_with('[') {
            return None;
        }
        serde_json::from_str(trimmed).ok()
    }

    /// Serializes a list of calls to the OpenAI-style JSON array.
    pub fn list_to_string(calls: &[ToolCall]) -> String {
        serde_json::to_string(calls).unwrap_or_else(|_| "[]".into())
    }
}

impl From<FunctionCallResponse> for ToolCall {
    fn from(r: FunctionCallResponse) -> Self {
        Self {
            id: r.id,
            name: r.function.name,
            arguments: r.function.arguments,
        }
    }
}

impl From<ToolCall> for FunctionCallResponse {
    fn from(c: ToolCall) -> Self {
        FunctionCallResponse {
            id: c.id,
            type_field: "function".into(),
            function: FunctionDetail {
                name: c.name,
                arguments: c.arguments,
            },
        }
    }
}

impl Serialize for ToolCall {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        FunctionCallResponse::from(self.clone()).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ToolCall {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        FunctionCallResponse::deserialize(deserializer).map(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_to_the_openai_wire_format() {
        let call = ToolCall::new("call_1", "weather", r#"{"city":"Lima"}"#);
        let v = serde_json::to_value(&call).unwrap();
        assert_eq!(
            v,
            json!({"id": "call_1", "type": "function",
                   "function": {"name": "weather", "arguments": "{\"city\":\"Lima\"}"}})
        );
        let back: ToolCall = serde_json::from_value(v).unwrap();
        assert_eq!(back, call);
    }

    #[test]
    fn parse_list_accepts_arrays_only() {
        let calls = ToolCall::parse_list(
            r#" [{"id":"a","type":"function","function":{"name":"f","arguments":"{}"}}]"#,
        )
        .unwrap();
        assert_eq!(calls, vec![ToolCall::new("a", "f", "{}")]);
        assert_eq!(ToolCall::parse_list("[]"), Some(vec![]));
        assert_eq!(ToolCall::parse_list("Hello"), None);
        assert_eq!(ToolCall::parse_list("[1, 2]"), None);
        assert_eq!(ToolCall::parse_list(r#"{"id":"a"}"#), None);
    }

    #[test]
    fn list_round_trip() {
        let calls = vec![
            ToolCall::from_value("a", "f", &json!({"x": 1})),
            ToolCall::new("b", "g", ""),
        ];
        let s = ToolCall::list_to_string(&calls);
        assert_eq!(ToolCall::parse_list(&s).unwrap(), calls);
    }

    #[test]
    fn arguments_value_normalizes_input() {
        assert_eq!(ToolCall::new("", "f", "").arguments_value(), json!({}));
        assert_eq!(
            ToolCall::new("", "f", r#"{"a":1}"#).arguments_value(),
            json!({"a": 1})
        );
        assert_eq!(
            ToolCall::new("", "f", "[1]").arguments_value(),
            json!({"input": [1]})
        );
        assert_eq!(
            ToolCall::new("", "f", "plain text").arguments_value(),
            json!({"input": "plain text"})
        );
    }
}
