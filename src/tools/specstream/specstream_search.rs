//! Web search through [SpecStream](https://specstream.ai/search-api).
//!
//! Each call is one `GET /v1/web-search` on the fast tier with page passages, so every
//! result carries the passages of its page closest to the query. The key comes from
//! `SPECSTREAM_API_KEY` or [`with_api_key`](SpecStreamSearch::with_api_key) and is sent
//! as a Bearer token.
//!
//! ```no_run
//! use langchain_rust::tools::{SpecStreamSearch, Tool};
//! use serde_json::json;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Reads SPECSTREAM_API_KEY.
//! let specstream = SpecStreamSearch::new().with_max_results(3)?;
//! println!("{}", specstream.run(json!("tokio release notes")).await?);
//! # Ok(())
//! # }
//! ```

use std::{error::Error, time::Duration};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::tools::Tool;

pub const SPECSTREAM_API_BASE: &str = "https://api.specstream.ai";
pub const DEFAULT_SPECSTREAM_MAX_RESULTS: u32 = 5;
const MAX_RESULTS: u32 = 100;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_ERROR_CHARS: usize = 300;

#[derive(Debug, Deserialize)]
struct SearchResponse {
    #[serde(default)]
    data: Option<SearchData>,
}

#[derive(Debug, Deserialize)]
struct SearchData {
    #[serde(default)]
    sources: Vec<SpecStreamResult>,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: Option<ApiError>,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    #[serde(default)]
    code: Option<String>,
    #[serde(default)]
    message: Option<String>,
}

/// One search result.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SpecStreamResult {
    #[serde(default)]
    pub title: Option<String>,
    pub url: String,
    /// The start of the page text.
    #[serde(default)]
    pub snippet: Option<String>,
    /// The passages of the page closest to the query.
    #[serde(default)]
    pub highlights: Vec<String>,
    /// The date the page states it was published, when it states one.
    #[serde(default)]
    pub published: Option<String>,
}

/// Web search tool backed by SpecStream's search API.
///
/// Tool output: numbered `Title / URL` blocks followed by the page's passages, or its
/// snippet when it has none. No results gives a plain message instead of an error. Use
/// [`search`](Self::search) for typed results.
#[derive(Debug, Clone)]
pub struct SpecStreamSearch {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    max_results: u32,
}

impl Default for SpecStreamSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl SpecStreamSearch {
    /// Reads the key from `SPECSTREAM_API_KEY`.
    pub fn new() -> Self {
        Self {
            client: build_client(DEFAULT_TIMEOUT),
            base_url: SPECSTREAM_API_BASE.to_string(),
            api_key: std::env::var("SPECSTREAM_API_KEY")
                .ok()
                .filter(|k| !k.trim().is_empty()),
            max_results: DEFAULT_SPECSTREAM_MAX_RESULTS,
        }
    }

    /// Use this key instead of `SPECSTREAM_API_KEY`.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        self.api_key = (!api_key.trim().is_empty()).then_some(api_key);
        self
    }

    /// API origin, `https://api.specstream.ai` by default. Mostly for tests and proxies.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Request timeout, 30 s by default.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = build_client(timeout);
        self
    }

    /// Results per query, 1 to 100 (default 5).
    pub fn with_max_results(mut self, max_results: u32) -> Result<Self, Box<dyn Error>> {
        if !(1..=MAX_RESULTS).contains(&max_results) {
            return Err(format!("max_results must be 1-{MAX_RESULTS}, got {max_results}").into());
        }
        self.max_results = max_results;
        Ok(self)
    }

    /// Runs one search.
    pub async fn search(&self, query: &str) -> Result<Vec<SpecStreamResult>, Box<dyn Error>> {
        if query.trim().is_empty() {
            return Err("query cannot be empty".into());
        }
        let key = self
            .api_key
            .as_deref()
            .ok_or("SpecStream needs an API key: set SPECSTREAM_API_KEY or use with_api_key")?;
        let response = self
            .client
            .get(format!("{}/v1/web-search", self.base_url))
            .bearer_auth(key)
            .query(&[
                ("q", query.to_string()),
                ("num", self.max_results.to_string()),
                ("speed", "fast".to_string()),
                ("passages", "true".to_string()),
            ])
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            return Err(
                format!("SpecStream search failed ({status}): {}", error_text(&text)).into(),
            );
        }
        let response: SearchResponse = response.json().await?;
        Ok(response.data.map(|data| data.sources).unwrap_or_default())
    }
}

fn build_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("failed to build reqwest client")
}

/// The API's own `code: message` when the body is its JSON error, else the cut body.
fn error_text(body: &str) -> String {
    let parsed = serde_json::from_str::<ErrorResponse>(body)
        .ok()
        .and_then(|r| r.error);
    let text = match parsed {
        Some(ApiError {
            code: Some(code),
            message: Some(message),
        }) => format!("{code}: {message}"),
        Some(ApiError {
            message: Some(message),
            ..
        }) => message,
        _ => body.to_string(),
    };
    text.chars().take(MAX_ERROR_CHARS).collect()
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|v| !v.is_empty())
}

fn format_results(results: &[SpecStreamResult]) -> String {
    results
        .iter()
        .enumerate()
        .map(|(i, result)| {
            let title = non_empty(result.title.as_deref()).unwrap_or("(untitled)");
            let mut text = format!("[{}] Title: {title}\nURL: {}", i + 1, result.url);
            if let Some(published) = non_empty(result.published.as_deref()) {
                text.push_str(&format!("\nPublished: {published}"));
            }
            let passages: Vec<&str> = result
                .highlights
                .iter()
                .map(|h| h.trim())
                .filter(|h| !h.is_empty())
                .collect();
            if !passages.is_empty() {
                text.push_str(&format!("\nPassages: {}", passages.join(" ... ")));
            } else if let Some(snippet) = non_empty(result.snippet.as_deref()) {
                text.push_str(&format!("\nSnippet: {snippet}"));
            }
            text
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[async_trait]
impl Tool for SpecStreamSearch {
    fn name(&self) -> String {
        "specstream-search".to_string()
    }

    fn description(&self) -> String {
        "Searches the web with SpecStream and returns ranked results with title, URL and the \
         passages of each page closest to the query. Useful for current events and facts \
         that need a source. Input should be a search query."
            .to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query"
                }
            },
            "required": ["query"]
        })
    }

    /// Accepts a raw query, `{"query": ...}` (tool calling) or `{"input": ...}`.
    async fn parse_input(&self, input: &str) -> Value {
        match serde_json::from_str::<Value>(input) {
            Ok(value @ Value::Object(_)) => value,
            _ => Value::String(input.to_string()),
        }
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let query = match &input {
            Value::String(query) => query.as_str(),
            Value::Object(map) => map
                .get("query")
                .or_else(|| map.get("input"))
                .and_then(Value::as_str)
                .ok_or("expected a string or an object with a `query` field")?,
            _ => return Err("expected a string or an object with a `query` field".into()),
        };
        let results = self.search(query).await?;
        if results.is_empty() {
            return Ok(format!("No results found for query: {query}"));
        }
        Ok(format_results(&results))
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
