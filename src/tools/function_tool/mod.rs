use std::{error::Error, future::Future, marker::PhantomData};

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde_json::{json, Value};

use super::Tool;

/// A [`Tool`] built from an async function, without writing a struct and a trait impl.
///
/// The function takes any `serde` type: the arguments the model sends are deserialized
/// into it. `parameters` is the JSON schema advertised to the model; it should describe
/// that type.
///
/// ```rust,ignore
/// #[derive(serde::Deserialize)]
/// struct AddInput { a: f64, b: f64 }
///
/// let adder = FunctionTool::new(
///     "adder",
///     "Adds two numbers together",
///     json!({
///         "type": "object",
///         "properties": { "a": {"type": "number"}, "b": {"type": "number"} },
///         "required": ["a", "b"]
///     }),
///     |input: AddInput| async move {
///         Ok::<_, String>(format!("The sum of {} and {} is {}", input.a, input.b, input.a + input.b))
///     },
/// );
/// let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(adder)];
/// ```
///
/// For a tool that takes a single string, [`FunctionTool::text`] fills in the schema.
///
/// Input that doesn't deserialize comes back to the model as a tool error naming the
/// expected schema, so it can retry; the function isn't called.
pub struct FunctionTool<T, F> {
    name: String,
    description: String,
    parameters: Value,
    func: F,
    // `fn() -> T` keeps the tool Send + Sync whatever `T` is; no `T` is ever stored.
    _input: PhantomData<fn() -> T>,
}

impl<T, F> FunctionTool<T, F> {
    pub fn new<N: Into<String>, D: Into<String>>(
        name: N,
        description: D,
        parameters: Value,
        func: F,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            func,
            _input: PhantomData,
        }
    }
}

impl<F> FunctionTool<String, F> {
    /// A tool whose function takes the model's input as one string. The advertised schema
    /// is `{"input": string}`, described by `description`.
    pub fn text<N: Into<String>, D: Into<String>>(name: N, description: D, func: F) -> Self {
        let description = description.into();
        let parameters = json!({
            "type": "object",
            "properties": {
                "input": { "type": "string", "description": description }
            },
            "required": ["input"]
        });
        Self::new(name, description, parameters, func)
    }
}

impl<T: DeserializeOwned, F> FunctionTool<T, F> {
    /// Deserializes the model's arguments. Besides the value itself, accepts the
    /// `{"input": ...}` wrapper the default schema asks for, arguments double-encoded as a
    /// JSON string, and a bare value for a string input (e.g. `42` when the model skipped
    /// the quotes).
    fn decode(&self, value: Value) -> Result<T, Box<dyn Error>> {
        let first_err = match serde_json::from_value::<T>(value.clone()) {
            Ok(input) => return Ok(input),
            Err(e) => e,
        };
        let fallbacks = [
            value.get("input").cloned(),
            value.as_str().and_then(|s| serde_json::from_str(s).ok()),
            (!value.is_string()).then(|| Value::String(value.to_string())),
        ];
        fallbacks
            .into_iter()
            .flatten()
            .find_map(|candidate| serde_json::from_value::<T>(candidate).ok())
            .ok_or_else(|| {
                format!(
                    "invalid input for tool `{}`: {first_err}. Expected arguments matching: {}",
                    self.name, self.parameters
                )
                .into()
            })
    }
}

#[async_trait]
impl<T, F, Fut, O, E> Tool for FunctionTool<T, F>
where
    T: DeserializeOwned + Send + 'static,
    F: Fn(T) -> Fut + Send + Sync,
    Fut: Future<Output = Result<O, E>> + Send,
    O: ToString,
    E: Into<Box<dyn Error>>,
{
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn parameters(&self) -> Value {
        self.parameters.clone()
    }

    /// Keeps the arguments as JSON (the default turns objects into a string); anything
    /// that isn't JSON is passed on as a string.
    async fn parse_input(&self, input: &str) -> Value {
        serde_json::from_str(input).unwrap_or_else(|_| Value::String(input.to_string()))
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let input = self.decode(input)?;
        (self.func)(input)
            .await
            .map(|output| output.to_string())
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests;
