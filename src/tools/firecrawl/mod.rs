//! Web search and page scraping through [Firecrawl](https://docs.firecrawl.dev).
//!
//! [`FirecrawlSearch`] searches the web and can return each result's page as markdown in
//! the same call. [`FirecrawlScrape`] turns one URL into clean markdown, with JavaScript
//! rendering and PDF parsing done on Firecrawl's side. Both read `FIRECRAWL_API_KEY`.

use std::{error::Error, time::Duration};

use serde::{de::DeserializeOwned, Deserialize};
use serde_json::{Map, Value};

mod firecrawl_scrape;
mod firecrawl_search;
pub use firecrawl_scrape::*;
pub use firecrawl_search::*;

pub const FIRECRAWL_API_BASE: &str = "https://api.firecrawl.dev";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_ERROR_BODY_CHARS: usize = 500;
/// Tells Firecrawl which client sent the request. Custom integration names start with `_`.
const ORIGIN: &str = "langchain-rust";
const INTEGRATION: &str = "_langchain-rust";

/// Page metadata Firecrawl returns with scraped content.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FirecrawlMetadata {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    /// The URL that was requested.
    #[serde(default, rename = "sourceURL")]
    pub source_url: Option<String>,
    /// The URL after redirects.
    #[serde(default)]
    pub url: Option<String>,
    /// HTTP status of the target page (Firecrawl itself answers 200).
    #[serde(default)]
    pub status_code: Option<u16>,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Envelope<T> {
    #[serde(default)]
    success: bool,
    data: Option<T>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    warning: Option<String>,
}

/// HTTP settings both tools share.
#[derive(Debug, Clone)]
struct FirecrawlClient {
    http: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl FirecrawlClient {
    fn new() -> Self {
        Self {
            http: build_client(DEFAULT_TIMEOUT),
            base_url: FIRECRAWL_API_BASE.to_string(),
            api_key: std::env::var("FIRECRAWL_API_KEY").unwrap_or_default(),
        }
    }

    fn set_base_url(&mut self, base_url: String) {
        self.base_url = base_url.trim_end_matches('/').to_string();
    }

    /// POSTs `body` plus the origin and integration fields, and returns `data`.
    async fn post<T: DeserializeOwned>(
        &self,
        path: &str,
        mut body: Map<String, Value>,
    ) -> Result<T, Box<dyn Error>> {
        if self.api_key.is_empty() {
            return Err("FIRECRAWL_API_KEY is not set (or use with_api_key)".into());
        }
        body.insert("origin".into(), ORIGIN.into());
        body.insert("integration".into(), INTEGRATION.into());
        let response = self
            .http
            .post(format!("{}{path}", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let text: String = text.chars().take(MAX_ERROR_BODY_CHARS).collect();
            return Err(format!("Firecrawl request failed ({status}): {text}").into());
        }
        let envelope: Envelope<T> = response.json().await?;
        if let Some(warning) = &envelope.warning {
            tracing::warn!(%warning, "firecrawl warning");
        }
        match envelope {
            Envelope {
                success: true,
                data: Some(data),
                ..
            } => Ok(data),
            Envelope { error, .. } => Err(format!(
                "Firecrawl request failed: {}",
                error.unwrap_or_else(|| "no data in response".into())
            )
            .into()),
        }
    }
}

fn build_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("failed to build reqwest client")
}

/// Accepts a raw string, `{"<field>": ...}` (tool calling) or `{"input": ...}`.
fn input_field<'a>(input: &'a Value, field: &str) -> Result<&'a str, Box<dyn Error>> {
    let expected = || format!("expected a string or an object with a `{field}` field");
    match input {
        Value::String(value) => Ok(value.as_str()),
        Value::Object(map) => map
            .get(field)
            .or_else(|| map.get("input"))
            .and_then(Value::as_str)
            .ok_or_else(|| expected().into()),
        _ => Err(expected().into()),
    }
}

/// The trimmed string, or `None` when it is missing or blank.
fn non_empty(s: Option<&String>) -> Option<&str> {
    s.map(|s| s.trim()).filter(|s| !s.is_empty())
}

/// Cuts `text` to `max` chars. A cut ends with ` [truncated]`, as in `WebScrapper`.
fn truncate(text: &str, max: Option<usize>) -> String {
    match max {
        Some(max) if text.chars().count() > max => {
            let cut: String = text.chars().take(max).collect();
            format!("{cut} [truncated]")
        }
        _ => text.to_string(),
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
