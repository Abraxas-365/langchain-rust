use std::collections::HashMap;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use crate::{
    chain::{Chain, ChainError, ChainEventStream, DEFAULT_OUTPUT_KEY, DEFAULT_RESULT_KEY},
    language_models::{llm::LLM, GenerateResult, LLMEvent, TokenUsage},
    observability::traced,
    prompt::PromptArgs,
    schemas::{messages::value_to_text, Message},
};

/// Output key of [`RouterChain::execute`] holding the chosen destination's name
/// (`DEFAULT` when the default chain ran).
pub const ROUTER_DESTINATION_KEY: &str = "destination";
/// What the router answers when no destination fits.
pub const DEFAULT_DESTINATION: &str = "DEFAULT";
const ROUTER_DEFAULT_INPUT_KEY: &str = "input";

const ROUTER_PROMPT: &str = "Given a raw text input, select the destination best suited for it. \
You will be given the names of the available destinations and a description of what each one \
is good for.

<< DESTINATIONS >>
{destinations}

Answer with a JSON object and nothing else:
{\"destination\": \"<one of the destination names above, or DEFAULT if none of them fits>\"}";

struct Destination {
    name: String,
    description: String,
    chain: Box<dyn Chain>,
}

/// The router's decision.
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    /// The chosen destination; `None` for the default chain.
    pub destination: Option<String>,
    /// Usage of the routing call.
    pub tokens: Option<TokenUsage>,
}

/// Asks an LLM which of several named chains should handle the input, then runs that
/// chain with the same input. If none fits (or the LLM names one that doesn't exist), the
/// default chain runs; without a default that's an error.
///
/// For routing by embeddings instead of an LLM call, see
/// [`semantic_router`](crate::semantic_router).
///
/// ```rust,ignore
/// let router = RouterChainBuilder::new()
///     .llm(OpenAI::default())
///     .destination("physics", "Questions about physics", physics_chain)
///     .destination("math", "Math problems and proofs", math_chain)
///     .default_chain(general_chain)
///     .build()?;
/// let answer = router.invoke(prompt_args! { "input" => "What is a black hole?" }).await?;
/// ```
pub struct RouterChain {
    llm: Box<dyn LLM>,
    destinations: Vec<Destination>,
    default_chain: Option<Box<dyn Chain>>,
    input_key: String,
    output_key: String,
}

pub struct RouterChainBuilder {
    llm: Option<Box<dyn LLM>>,
    destinations: Vec<Destination>,
    default_chain: Option<Box<dyn Chain>>,
    input_key: String,
    output_key: String,
}

impl Default for RouterChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RouterChainBuilder {
    pub fn new() -> Self {
        Self {
            llm: None,
            destinations: Vec::new(),
            default_chain: None,
            input_key: ROUTER_DEFAULT_INPUT_KEY.to_string(),
            output_key: DEFAULT_OUTPUT_KEY.to_string(),
        }
    }

    /// The LLM that picks the destination.
    pub fn llm<L: Into<Box<dyn LLM>>>(mut self, llm: L) -> Self {
        self.llm = Some(llm.into());
        self
    }

    /// Adds a destination. The LLM sees `name` and `description`, so make the description
    /// say what inputs the chain is good for.
    pub fn destination<C: Into<Box<dyn Chain>>>(
        mut self,
        name: &str,
        description: &str,
        chain: C,
    ) -> Self {
        self.destinations.push(Destination {
            name: name.trim().to_string(),
            description: description.trim().to_string(),
            chain: chain.into(),
        });
        self
    }

    /// Runs when no destination fits.
    pub fn default_chain<C: Into<Box<dyn Chain>>>(mut self, chain: C) -> Self {
        self.default_chain = Some(chain.into());
        self
    }

    /// Input key holding the text to route (default `input`). Every input is forwarded
    /// to the destination as is.
    pub fn input_key(mut self, key: &str) -> Self {
        self.input_key = key.to_string();
        self
    }

    /// Output key of the answer in [`execute`](Chain::execute) (default `output`).
    pub fn output_key(mut self, key: &str) -> Self {
        self.output_key = key.to_string();
        self
    }

    pub fn build(self) -> Result<RouterChain, ChainError> {
        let llm = self
            .llm
            .ok_or_else(|| ChainError::MissingObject("llm".into()))?;
        if self.destinations.is_empty() {
            return Err(ChainError::MissingObject("destination".into()));
        }
        for (i, dest) in self.destinations.iter().enumerate() {
            if dest.name.is_empty() || dest.name.eq_ignore_ascii_case(DEFAULT_DESTINATION) {
                return Err(ChainError::OtherError(format!(
                    "invalid router destination name {:?}",
                    dest.name
                )));
            }
            if self.destinations[..i]
                .iter()
                .any(|d| d.name.eq_ignore_ascii_case(&dest.name))
            {
                return Err(ChainError::OtherError(format!(
                    "duplicate router destination {:?}",
                    dest.name
                )));
            }
        }
        Ok(RouterChain {
            llm,
            destinations: self.destinations,
            default_chain: self.default_chain,
            input_key: self.input_key,
            output_key: self.output_key,
        })
    }
}

/// The destination name in the router's answer: a JSON object (possibly fenced or
/// surrounded by text), or failing that the bare answer.
pub(super) fn parse_destination(answer: &str) -> String {
    if let (Some(start), Some(end)) = (answer.find('{'), answer.rfind('}')) {
        if start < end {
            if let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(&answer[start..=end]) {
                if let Some(Value::String(name)) = obj.get("destination") {
                    return name.trim().to_string();
                }
            }
        }
    }
    answer
        .trim()
        .trim_matches(|c: char| c == '"' || c == '\'' || c == '`' || c == '.')
        .trim()
        .to_string()
}

fn add_tokens(a: Option<TokenUsage>, b: Option<&TokenUsage>) -> Option<TokenUsage> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.sum(b)),
        (a, b) => a.or_else(|| b.cloned()),
    }
}

impl RouterChain {
    pub(super) fn prompt(&self, input: &str) -> Vec<Message> {
        let destinations = self
            .destinations
            .iter()
            .map(|d| format!("{}: {}", d.name, d.description))
            .collect::<Vec<_>>()
            .join("\n");
        vec![
            Message::new_system_message(ROUTER_PROMPT.replace("{destinations}", &destinations)),
            Message::new_human_message(input),
        ]
    }

    /// Asks the LLM where `input_variables` should go, without running anything else.
    pub async fn route(&self, input_variables: &PromptArgs) -> Result<Route, ChainError> {
        let input = input_variables
            .get(&self.input_key)
            .map(value_to_text)
            .ok_or_else(|| ChainError::MissingInputVariable(self.input_key.clone()))?;
        let result = self.llm.generate(&self.prompt(&input)).await?;
        let name = parse_destination(&result.generation);
        let destination = self
            .destinations
            .iter()
            .find(|d| d.name.eq_ignore_ascii_case(&name))
            .map(|d| d.name.clone());
        if destination.is_none() && !name.eq_ignore_ascii_case(DEFAULT_DESTINATION) {
            tracing::warn!(answer = %result.generation, "router answered an unknown destination");
            if self.default_chain.is_none() {
                return Err(ChainError::OtherError(format!(
                    "router picked unknown destination {name:?} and there is no default chain; \
                     expected one of {:?}",
                    self.destinations
                        .iter()
                        .map(|d| &d.name)
                        .collect::<Vec<_>>()
                )));
            }
        }
        Ok(Route {
            destination,
            tokens: result.tokens,
        })
    }

    fn chain_for(&self, route: &Route) -> Result<&dyn Chain, ChainError> {
        match &route.destination {
            Some(name) => Ok(self
                .destinations
                .iter()
                .find(|d| &d.name == name)
                .map(|d| d.chain.as_ref())
                .expect("route() only returns known destinations")),
            None => self.default_chain.as_deref().ok_or_else(|| {
                ChainError::OtherError("no destination fits and there is no default chain".into())
            }),
        }
    }

    fn span(&self) -> tracing::Span {
        tracing::info_span!(
            "chain.call",
            chain = "router",
            destination = tracing::field::Empty,
            error = tracing::field::Empty
        )
    }
}

fn destination_label(route: &Route) -> &str {
    route.destination.as_deref().unwrap_or(DEFAULT_DESTINATION)
}

#[async_trait]
impl Chain for RouterChain {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        let span = self.span();
        traced(
            span.clone(),
            async {
                let route = self.route(&input_variables).await?;
                span.record("destination", destination_label(&route));
                let mut result = self.chain_for(&route)?.call(input_variables).await?;
                result.tokens = add_tokens(result.tokens, route.tokens.as_ref());
                Ok(result)
            },
            |_, _| {},
        )
        .await
    }

    /// The destination's own outputs, plus the answer under the output key, the full
    /// result (routing tokens included) and [`ROUTER_DESTINATION_KEY`].
    async fn execute(
        &self,
        input_variables: PromptArgs,
    ) -> Result<HashMap<String, Value>, ChainError> {
        let span = self.span();
        traced(
            span.clone(),
            async {
                let route = self.route(&input_variables).await?;
                span.record("destination", destination_label(&route));
                let chain = self.chain_for(&route)?;
                let mut output = chain.execute(input_variables).await?;
                let mut result: GenerateResult = match output.get(DEFAULT_RESULT_KEY) {
                    Some(result) => serde_json::from_value(result.clone())?,
                    None => {
                        let key = chain
                            .get_output_keys()
                            .first()
                            .cloned()
                            .unwrap_or_else(|| DEFAULT_OUTPUT_KEY.to_string());
                        GenerateResult {
                            generation: output.get(&key).map(value_to_text).unwrap_or_default(),
                            ..Default::default()
                        }
                    }
                };
                result.tokens = add_tokens(result.tokens, route.tokens.as_ref());
                output.insert(self.output_key.clone(), json!(result.generation));
                output.insert(DEFAULT_RESULT_KEY.to_string(), json!(result));
                output.insert(
                    ROUTER_DESTINATION_KEY.to_string(),
                    json!(destination_label(&route)),
                );
                Ok(output)
            },
            |_, _| {},
        )
        .await
    }

    /// Routes, then streams the destination's answer. The final `Done` includes the
    /// routing tokens.
    fn call_stream<'a>(&'a self, input_variables: PromptArgs) -> ChainEventStream<'a> {
        Box::pin(
            futures::stream::once(async move {
                let routed = async {
                    let route = self.route(&input_variables).await?;
                    let chain = self.chain_for(&route)?;
                    Ok::<_, ChainError>((route, chain))
                };
                match routed.await {
                    Ok((route, chain)) => {
                        let tokens = route.tokens;
                        Box::pin(chain.call_stream(input_variables).map(move |event| {
                            event.map(|event| match event {
                                LLMEvent::Done(mut result) => {
                                    result.tokens = add_tokens(result.tokens, tokens.as_ref());
                                    LLMEvent::Done(result)
                                }
                                other => other,
                            })
                        })) as ChainEventStream<'a>
                    }
                    Err(e) => Box::pin(futures::stream::iter([Err(e)])) as ChainEventStream<'a>,
                }
            })
            .flatten(),
        )
    }

    fn get_input_keys(&self) -> Vec<String> {
        vec![self.input_key.clone()]
    }

    fn get_output_keys(&self) -> Vec<String> {
        vec![
            self.output_key.clone(),
            DEFAULT_RESULT_KEY.to_string(),
            ROUTER_DESTINATION_KEY.to_string(),
        ]
    }
}
