//! Web search through [Keenable](https://docs.keenable.ai).
//!
//! Works without an API key: keyless calls go to `POST /v1/search/public` and name the
//! calling app in the `X-Keenable-Title` header. With a key (`KEENABLE_API_KEY` or
//! [`with_api_key`](KeenableSearch::with_api_key)) calls go to `POST /v1/search` with
//! `X-API-Key` instead.
//!
//! ```no_run
//! use langchain_rust::tools::{KeenableSearch, Tool};
//! use serde_json::json;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // No key needed; KEENABLE_API_KEY is used if set.
//! let keenable = KeenableSearch::new().with_max_results(3)?;
//! println!("{}", keenable.run(json!("tokio release notes")).await?);
//! # Ok(())
//! # }
//! ```

use std::{error::Error, time::Duration};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::tools::Tool;

pub const KEENABLE_API_BASE: &str = "https://api.keenable.ai";
pub const DEFAULT_KEENABLE_MAX_RESULTS: u32 = 5;
/// Sent as `X-Keenable-Title` on keyless calls unless overridden.
pub const DEFAULT_KEENABLE_APP_TITLE: &str = "langchain-rust";
const MAX_RESULTS: u32 = 50;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
const MAX_ERROR_BODY_CHARS: usize = 500;

#[derive(Debug, Deserialize)]
struct SearchResponse {
    #[serde(default)]
    results: Vec<KeenableResult>,
}

/// One search result.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct KeenableResult {
    #[serde(default)]
    pub title: Option<String>,
    pub url: String,
    #[serde(default)]
    pub description: Option<String>,
    /// The passage of the page most relevant to the query.
    #[serde(default)]
    pub snippet: Option<String>,
    /// When Keenable last crawled the page (RFC 3339).
    #[serde(default)]
    pub acquired_at: Option<String>,
}

/// Web search tool backed by Keenable's search API.
///
/// Tool output: numbered `Title / URL` blocks followed by the snippet. No results gives a
/// plain message instead of an error. Use [`search`](Self::search) for typed results.
#[derive(Debug, Clone)]
pub struct KeenableSearch {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    app_title: String,
    max_results: u32,
}

impl Default for KeenableSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl KeenableSearch {
    /// Keyless unless `KEENABLE_API_KEY` is set.
    pub fn new() -> Self {
        Self {
            client: build_client(DEFAULT_TIMEOUT),
            base_url: KEENABLE_API_BASE.to_string(),
            api_key: std::env::var("KEENABLE_API_KEY")
                .ok()
                .filter(|k| !k.trim().is_empty()),
            app_title: DEFAULT_KEENABLE_APP_TITLE.to_string(),
            max_results: DEFAULT_KEENABLE_MAX_RESULTS,
        }
    }

    /// Use the keyed endpoint with this key. An empty key switches back to keyless.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        let api_key = api_key.into();
        self.api_key = (!api_key.trim().is_empty()).then_some(api_key);
        self
    }

    /// App name sent as `X-Keenable-Title` on keyless calls (`langchain-rust` by default).
    pub fn with_app_title(mut self, app_title: impl Into<String>) -> Self {
        self.app_title = app_title.into();
        self
    }

    /// API origin, `https://api.keenable.ai` by default. Mostly for tests and proxies.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Request timeout, 30 s by default.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = build_client(timeout);
        self
    }

    /// Results per query, 1 to 50 (default 5).
    pub fn with_max_results(mut self, max_results: u32) -> Result<Self, Box<dyn Error>> {
        if !(1..=MAX_RESULTS).contains(&max_results) {
            return Err(format!("max_results must be 1-{MAX_RESULTS}, got {max_results}").into());
        }
        self.max_results = max_results;
        Ok(self)
    }

    /// Runs one search.
    pub async fn search(&self, query: &str) -> Result<Vec<KeenableResult>, Box<dyn Error>> {
        if query.trim().is_empty() {
            return Err("query cannot be empty".into());
        }
        let body = json!({ "query": query, "max_results": self.max_results });
        let request = match &self.api_key {
            Some(key) => self
                .client
                .post(format!("{}/v1/search", self.base_url))
                .header("X-API-Key", key),
            None => self
                .client
                .post(format!("{}/v1/search/public", self.base_url))
                .header("X-Keenable-Title", &self.app_title),
        };
        let response = request.json(&body).send().await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let text: String = text.chars().take(MAX_ERROR_BODY_CHARS).collect();
            return Err(format!("Keenable search failed ({status}): {text}").into());
        }
        let response: SearchResponse = response.json().await?;
        Ok(response.results)
    }
}

fn build_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("failed to build reqwest client")
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|v| !v.is_empty())
}

fn format_results(results: &[KeenableResult]) -> String {
    results
        .iter()
        .enumerate()
        .map(|(i, result)| {
            let title = non_empty(result.title.as_deref()).unwrap_or("(untitled)");
            let mut text = format!("[{}] Title: {title}\nURL: {}", i + 1, result.url);
            if let Some(description) = non_empty(result.description.as_deref()) {
                text.push_str(&format!("\nDescription: {description}"));
            }
            if let Some(snippet) = non_empty(result.snippet.as_deref()) {
                text.push_str(&format!("\nSnippet: {snippet}"));
            }
            text
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[async_trait]
impl Tool for KeenableSearch {
    fn name(&self) -> String {
        "keenable-search".to_string()
    }

    fn description(&self) -> String {
        "Searches the web with Keenable and returns ranked results with title, URL and the \
         most relevant passage of each page. Useful for current events and facts that need \
         a source. Input should be a search query."
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
