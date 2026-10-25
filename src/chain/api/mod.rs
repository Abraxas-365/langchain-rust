//! API chain: the model writes an HTTP request from API docs, the chain sends it (only to
//! URLs you allow) and the model answers from the response.

use std::{collections::HashMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use reqwest::{redirect, Method};
use serde_json::{json, Value};
use url::Url;

use crate::{
    chain::{Chain, ChainError, LLMChainBuilder, DEFAULT_RESULT_KEY},
    language_models::{llm::LLM, GenerateResult, TokenUsage},
    output_parsers::extract_json,
    prompt::{PromptArgs, PromptTemplate, TemplateFormat},
};

/// Default input key of [`APIChain`], as in langchaingo.
pub const API_CHAIN_DEFAULT_INPUT_KEY: &str = "input";
/// Default output key of [`APIChain`], as in langchaingo.
pub const API_CHAIN_DEFAULT_OUTPUT_KEY: &str = "answer";
/// [`execute`](Chain::execute) output key with the request sent, e.g. `GET https://...`.
pub const API_CHAIN_REQUEST_KEY: &str = "api_request";
/// [`execute`](Chain::execute) output key with the response body (possibly truncated).
pub const API_CHAIN_RESPONSE_KEY: &str = "api_response";

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const DEFAULT_MAX_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_REDIRECTS: usize = 5;
const TRUNCATED: &str = "\n[response truncated]";

const REQUEST_PROMPT: &str = r#"Here is the documentation of an API:

{{api_docs}}

Write the API call that answers the question below. Use only URLs and parameters from the documentation, and leave out anything the question doesn't need.

Question: {{input}}

Reply with only a JSON object, with the body set to null when the call has none:
{"method": "GET", "url": "<full URL, with query parameters>", "body": null}"#;

const ANSWER_PROMPT: &str = r#"Question: {{input}}

To answer it, this API call was made:
{{api_request}}

The API responded:
{{api_response}}

Answer the question from this response. Be concise, and say so if the response doesn't answer it."#;

/// Answers questions with an HTTP API: the model writes the request from the API's
/// documentation, the chain sends it, and the model answers from the response. Like
/// LangChain's and langchaingo's `APIChain`.
///
/// The model chooses the URL, so the chain only calls the URLs you allow:
///
/// - Every request URL, and every redirect, must start with one of `allowed_urls`:
///   same scheme, host and port, and a path under the allowed path (`https://api.x.com/v1`
///   allows `/v1` and `/v1/...`, not `/v1x` or `/v2`). Only these URLs are reachable,
///   so a question can't make the chain call an internal service.
/// - Only `GET` is allowed unless you add methods with
///   [`with_allowed_methods`](Self::with_allowed_methods).
/// - Headers (e.g. an API key) come only from [`with_header`](Self::with_header), never
///   from the model.
/// - Requests time out after 30 s ([`with_timeout`](Self::with_timeout)) and at most 1 MiB
///   of the response is read ([`with_max_response_bytes`](Self::with_max_response_bytes));
///   a longer response is cut and marked `[response truncated]`.
/// - A non-2xx status is an error.
///
/// ```rust,ignore
/// let chain = APIChain::new(
///     OpenAI::default(),
///     OPEN_METEO_DOCS,
///     ["https://api.open-meteo.com/v1/"],
/// )?;
/// let answer = chain
///     .invoke(prompt_args! { "input" => "What's the temperature in Lima right now?" })
///     .await?;
/// ```
pub struct APIChain {
    request_chain: Box<dyn Chain>,
    answer_chain: Box<dyn Chain>,
    api_docs: String,
    allowed_urls: Arc<Vec<Url>>,
    allowed_methods: Vec<Method>,
    headers: Vec<(String, String)>,
    timeout: Duration,
    max_response_bytes: usize,
    client: reqwest::Client,
    input_key: String,
    output_key: String,
}

/// The request the model wrote.
#[derive(Debug)]
struct ApiRequest {
    method: Method,
    url: Url,
    body: Option<Value>,
}

impl APIChain {
    /// `api_docs` describes the API to the model; `allowed_urls` are the URL prefixes it
    /// may call (at least one).
    pub fn new<L, D, U>(llm: L, api_docs: D, allowed_urls: U) -> Result<Self, ChainError>
    where
        L: Into<Box<dyn LLM>>,
        D: Into<String>,
        U: IntoIterator,
        U::Item: AsRef<str>,
    {
        let llm = llm.into();
        let request_chain = prompt_chain(llm.clone_box(), REQUEST_PROMPT, &["api_docs", "input"]);
        let answer_chain = prompt_chain(
            llm,
            ANSWER_PROMPT,
            &["input", "api_request", "api_response"],
        );
        Self::from_chains(request_chain, answer_chain, api_docs, allowed_urls)
    }

    /// Uses your own chains. `request_chain` gets `api_docs` and `input` and must reply
    /// with the request as JSON (`{"method", "url", "body"}`); `answer_chain` gets `input`,
    /// `api_request` and `api_response`.
    pub fn from_chains<R, A, D, U>(
        request_chain: R,
        answer_chain: A,
        api_docs: D,
        allowed_urls: U,
    ) -> Result<Self, ChainError>
    where
        R: Into<Box<dyn Chain>>,
        A: Into<Box<dyn Chain>>,
        D: Into<String>,
        U: IntoIterator,
        U::Item: AsRef<str>,
    {
        let allowed_urls = allowed_urls
            .into_iter()
            .map(|u| parse_allowed_url(u.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        if allowed_urls.is_empty() {
            return Err(ChainError::MissingObject(
                "api chain: at least one allowed URL".into(),
            ));
        }
        let allowed_urls = Arc::new(allowed_urls);
        let client = client(allowed_urls.clone())?;
        Ok(Self {
            request_chain: request_chain.into(),
            answer_chain: answer_chain.into(),
            api_docs: api_docs.into(),
            allowed_urls,
            allowed_methods: vec![Method::GET],
            headers: Vec::new(),
            timeout: DEFAULT_TIMEOUT,
            max_response_bytes: DEFAULT_MAX_RESPONSE_BYTES,
            client,
            input_key: API_CHAIN_DEFAULT_INPUT_KEY.into(),
            output_key: API_CHAIN_DEFAULT_OUTPUT_KEY.into(),
        })
    }

    /// HTTP methods the model may use (default only `GET`). Replaces the list.
    pub fn with_allowed_methods<I>(mut self, methods: I) -> Result<Self, ChainError>
    where
        I: IntoIterator,
        I::Item: AsRef<str>,
    {
        self.allowed_methods = methods
            .into_iter()
            .map(|m| parse_method(m.as_ref()))
            .collect::<Result<_, _>>()?;
        Ok(self)
    }

    /// A header sent with every request, e.g. an API key. Not shown to the model.
    pub fn with_header<K: Into<String>, V: Into<String>>(mut self, name: K, value: V) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Request timeout, covering connect, redirects and reading the body (default 30 s).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The most bytes of the response body that are read and shown to the model
    /// (default 1 MiB).
    pub fn with_max_response_bytes(mut self, max: usize) -> Self {
        self.max_response_bytes = max;
        self
    }

    /// Input key with the question (default `input`). The prompts still get it as `input`.
    pub fn with_input_key<S: Into<String>>(mut self, key: S) -> Self {
        self.input_key = key.into();
        self
    }

    /// Key of the answer in [`execute`](Chain::execute)'s output (default `answer`).
    pub fn with_output_key<S: Into<String>>(mut self, key: S) -> Self {
        self.output_key = key.into();
        self
    }

    /// `(answer with the token usage of both calls, request line, response body)`.
    async fn run(
        &self,
        input_variables: PromptArgs,
    ) -> Result<(GenerateResult, String, String), ChainError> {
        let question = input_variables
            .get(&self.input_key)
            .cloned()
            .ok_or_else(|| ChainError::MissingInputVariable(self.input_key.clone()))?;

        let mut inputs = input_variables;
        inputs.remove(&self.input_key);
        inputs.insert("input".into(), question.clone());
        inputs.insert("api_docs".into(), json!(self.api_docs));
        let written = self.request_chain.call(inputs).await?;
        let request = self.parse_request(&written.generation)?;
        let request_line = format!("{} {}", request.method, request.url);
        let response = self.send(request).await?;

        let answer = self
            .answer_chain
            .call(HashMap::from([
                ("input".to_string(), question),
                ("api_request".to_string(), json!(request_line)),
                ("api_response".to_string(), json!(response)),
            ]))
            .await?;
        let tokens = sum_tokens(written.tokens, answer.tokens.clone());
        Ok((GenerateResult { tokens, ..answer }, request_line, response))
    }

    fn parse_request(&self, output: &str) -> Result<ApiRequest, ChainError> {
        let value =
            extract_json(output).map_err(|e| ChainError::OtherError(format!("api chain: {e}")))?;
        let method = match value.get("method") {
            None | Some(Value::Null) => Method::GET,
            Some(Value::String(m)) => parse_method(m)?,
            Some(other) => {
                return Err(ChainError::OtherError(format!(
                    "api chain: `method` must be a string, got {other}"
                )))
            }
        };
        let url = value.get("url").and_then(Value::as_str).ok_or_else(|| {
            ChainError::OtherError(format!("api chain: no `url` in the request: {value}"))
        })?;
        let url = Url::parse(url.trim())
            .map_err(|e| ChainError::OtherError(format!("api chain: invalid url {url:?}: {e}")))?;
        if !is_allowed(&self.allowed_urls, &url) {
            return Err(ChainError::OtherError(format!(
                "api chain: {url} is not an allowed URL"
            )));
        }
        if !self.allowed_methods.contains(&method) {
            return Err(ChainError::OtherError(format!(
                "api chain: method {method} is not allowed"
            )));
        }
        let body = value.get("body").filter(|b| !b.is_null()).cloned();
        Ok(ApiRequest { method, url, body })
    }

    async fn send(&self, request: ApiRequest) -> Result<String, ChainError> {
        let line = format!("{} {}", request.method, request.url);
        let fail = |e: String| ChainError::OtherError(format!("api chain: {line}: {e}"));

        let sends_body = !matches!(request.method, Method::GET | Method::HEAD);
        let mut builder = self
            .client
            .request(request.method, request.url)
            .timeout(self.timeout);
        for (name, value) in &self.headers {
            builder = builder.header(name, value);
        }
        if let (true, Some(body)) = (sends_body, &request.body) {
            builder = builder.json(body);
        }
        let mut response = builder.send().await.map_err(|e| fail(error_chain(&e)))?;

        let status = response.status();
        let mut body = Vec::new();
        let mut truncated = false;
        while let Some(chunk) = response.chunk().await.map_err(|e| fail(error_chain(&e)))? {
            let room = self.max_response_bytes - body.len();
            if chunk.len() > room {
                body.extend_from_slice(&chunk[..room]);
                truncated = true;
                break;
            }
            body.extend_from_slice(&chunk);
        }
        let mut text = String::from_utf8_lossy(&body).into_owned();
        if !status.is_success() {
            return Err(fail(format!("status {status}: {}", preview(&text))));
        }
        if truncated {
            text.push_str(TRUNCATED);
        }
        Ok(text)
    }
}

fn prompt_chain(llm: Box<dyn LLM>, template: &str, variables: &[&str]) -> Box<dyn Chain> {
    Box::new(
        LLMChainBuilder::new()
            .prompt(PromptTemplate::new(
                template.to_string(),
                variables.iter().map(|v| v.to_string()).collect(),
                TemplateFormat::Jinja2,
            ))
            .llm(llm)
            .build()
            .expect("prompt and llm are set"),
    )
}

/// A client that follows redirects only to allowed URLs.
fn client(allowed: Arc<Vec<Url>>) -> Result<reqwest::Client, ChainError> {
    let policy = redirect::Policy::custom(move |attempt| {
        if attempt.previous().len() > MAX_REDIRECTS {
            attempt.error("too many redirects")
        } else if is_allowed(&allowed, attempt.url()) {
            attempt.follow()
        } else {
            let msg = format!("redirect to {} is not an allowed URL", attempt.url());
            attempt.error(msg)
        }
    });
    reqwest::Client::builder()
        .redirect(policy)
        .build()
        .map_err(|e| ChainError::OtherError(format!("api chain: building the HTTP client: {e}")))
}

fn parse_allowed_url(raw: &str) -> Result<Url, ChainError> {
    let url = Url::parse(raw.trim()).map_err(|e| {
        ChainError::OtherError(format!("api chain: invalid allowed URL {raw:?}: {e}"))
    })?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(ChainError::OtherError(format!(
            "api chain: allowed URL {raw:?} must be http(s) with a host"
        )));
    }
    Ok(url)
}

fn parse_method(raw: &str) -> Result<Method, ChainError> {
    Method::from_bytes(raw.trim().to_ascii_uppercase().as_bytes())
        .map_err(|_| ChainError::OtherError(format!("api chain: invalid HTTP method {raw:?}")))
}

/// Same scheme, host and port as one of `allowed`, with a path at or under its path.
/// `Url` has already resolved `.`/`..` segments (also percent-encoded ones).
pub(crate) fn is_allowed(allowed: &[Url], url: &Url) -> bool {
    allowed.iter().any(|base| {
        let base_path = base.path().trim_end_matches('/');
        base.scheme() == url.scheme()
            && base.host_str() == url.host_str()
            && base.port_or_known_default() == url.port_or_known_default()
            && url.username().is_empty()
            && url.password().is_none()
            && (url.path() == base_path
                || url
                    .path()
                    .strip_prefix(base_path)
                    .is_some_and(|rest| rest.starts_with('/')))
    })
}

fn sum_tokens(a: Option<TokenUsage>, b: Option<TokenUsage>) -> Option<TokenUsage> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.sum(&b)),
        (a, b) => a.or(b),
    }
}

/// The error and its sources, so timeouts and connection errors are visible.
fn error_chain(err: &(dyn std::error::Error + 'static)) -> String {
    let mut msg = err.to_string();
    let mut source = err.source();
    while let Some(e) = source {
        msg.push_str(": ");
        msg.push_str(&e.to_string());
        source = e.source();
    }
    msg
}

fn preview(text: &str) -> String {
    const MAX: usize = 300;
    match text.char_indices().nth(MAX) {
        Some((cut, _)) => format!("{}...", &text[..cut]),
        None => text.to_string(),
    }
}

#[async_trait]
impl Chain for APIChain {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        Ok(self.run(input_variables).await?.0)
    }

    async fn execute(
        &self,
        input_variables: PromptArgs,
    ) -> Result<HashMap<String, Value>, ChainError> {
        let (result, request, response) = self.run(input_variables).await?;
        Ok(HashMap::from([
            (self.output_key.clone(), json!(result.generation)),
            (API_CHAIN_REQUEST_KEY.to_string(), json!(request)),
            (API_CHAIN_RESPONSE_KEY.to_string(), json!(response)),
            (DEFAULT_RESULT_KEY.to_string(), json!(result)),
        ]))
    }

    fn get_input_keys(&self) -> Vec<String> {
        vec![self.input_key.clone()]
    }

    fn get_output_keys(&self) -> Vec<String> {
        vec![
            self.output_key.clone(),
            API_CHAIN_REQUEST_KEY.to_string(),
            API_CHAIN_RESPONSE_KEY.to_string(),
            DEFAULT_RESULT_KEY.to_string(),
        ]
    }
}

#[cfg(test)]
mod tests;
