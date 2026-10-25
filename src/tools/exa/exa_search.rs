//! Web search through [Exa](https://exa.ai/docs/reference/search) (formerly Metaphor).
//!
//! Exa is a search engine built for LLMs: results come with highlights (the passages most
//! relevant to the query) or the page text, so the agent can answer from them without a
//! second fetch. langchaingo ships this as its `metaphor` tools.
//!
//! ```no_run
//! use langchain_rust::tools::{ExaContents, ExaSearch, Tool};
//! use serde_json::json;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Reads EXA_API_KEY.
//! let exa = ExaSearch::new()
//!     .with_num_results(3)?
//!     .with_category("news")
//!     .with_include_domains(["reuters.com", "apnews.com"])
//!     .with_contents(ExaContents::Highlights { max_characters: Some(1000) })?;
//! println!("{}", exa.run(json!("central bank rate decisions this week")).await?);
//! # Ok(())
//! # }
//! ```

use std::{error::Error, time::Duration};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::tools::Tool;

pub const EXA_API_BASE: &str = "https://api.exa.ai";
pub const DEFAULT_EXA_NUM_RESULTS: u32 = 5;
const MAX_NUM_RESULTS: u32 = 100;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const MAX_ERROR_BODY_CHARS: usize = 500;

/// What page content each result carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExaContents {
    /// Titles and URLs only.
    None,
    /// The passages Exa picks as most relevant to the query (the default), capped per
    /// result at `max_characters` if set.
    Highlights { max_characters: Option<u32> },
    /// The page text, capped at `max_characters` if set.
    Text { max_characters: Option<u32> },
}

impl Default for ExaContents {
    fn default() -> Self {
        Self::Highlights {
            max_characters: None,
        }
    }
}

impl ExaContents {
    fn to_json(&self) -> Option<Value> {
        let options = |max: &Option<u32>| match max {
            Some(max) => json!({ "maxCharacters": max }),
            None => json!(true),
        };
        match self {
            Self::None => None,
            Self::Highlights { max_characters } => {
                Some(json!({ "highlights": options(max_characters) }))
            }
            Self::Text { max_characters } => Some(json!({ "text": options(max_characters) })),
        }
    }

    fn check(&self) -> Result<(), Box<dyn Error>> {
        match self {
            Self::Highlights {
                max_characters: Some(0),
            }
            | Self::Text {
                max_characters: Some(0),
            } => Err("max_characters must be at least 1".into()),
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Deserialize)]
struct SearchResponse {
    results: Vec<ExaResult>,
    #[serde(default)]
    notices: Vec<Notice>,
    #[serde(default)]
    warnings: Vec<Value>,
}

/// Exa notices are meant to be shown to the model next to the results.
#[derive(Debug, Deserialize)]
struct Notice {
    #[serde(default)]
    message: String,
}

/// One search result. Content fields are set according to [`ExaContents`].
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExaResult {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub title: Option<String>,
    pub url: String,
    #[serde(default)]
    pub published_date: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub highlights: Vec<String>,
    #[serde(default)]
    pub text: Option<String>,
}

/// Exa results plus the notices Exa wants passed to the model.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExaSearchResults {
    pub results: Vec<ExaResult>,
    pub notices: Vec<String>,
}

/// Web search tool backed by Exa's `POST /search`.
///
/// Tool output: numbered `Title / URL / Published / Author` blocks followed by the
/// highlights or text, then any Exa notices. No results gives a plain message instead of
/// an error. Use [`search`](Self::search) for typed results.
#[derive(Debug, Clone)]
pub struct ExaSearch {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    num_results: u32,
    search_type: Option<String>,
    category: Option<String>,
    include_domains: Vec<String>,
    exclude_domains: Vec<String>,
    start_published_date: Option<String>,
    end_published_date: Option<String>,
    contents: ExaContents,
}

impl Default for ExaSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl ExaSearch {
    /// Reads the key from `EXA_API_KEY` (calls fail with a clear error if it is unset;
    /// see [`with_api_key`](Self::with_api_key)).
    pub fn new() -> Self {
        Self {
            client: build_client(DEFAULT_TIMEOUT),
            base_url: EXA_API_BASE.to_string(),
            api_key: std::env::var("EXA_API_KEY").unwrap_or_default(),
            num_results: DEFAULT_EXA_NUM_RESULTS,
            search_type: None,
            category: None,
            include_domains: Vec::new(),
            exclude_domains: Vec::new(),
            start_published_date: None,
            end_published_date: None,
            contents: ExaContents::default(),
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// API origin, `https://api.exa.ai` by default. Mostly for tests and proxies.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Request timeout, 60 s by default (deep searches take longer than plain ones).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = build_client(timeout);
        self
    }

    /// Results per query, 1 to 100 (default 5). Some search types allow fewer.
    pub fn with_num_results(mut self, num_results: u32) -> Result<Self, Box<dyn Error>> {
        if !(1..=MAX_NUM_RESULTS).contains(&num_results) {
            return Err(
                format!("num_results must be 1-{MAX_NUM_RESULTS}, got {num_results}").into(),
            );
        }
        self.num_results = num_results;
        Ok(self)
    }

    /// Search mode: `auto` (Exa's default), `lite`, `deep-lite`, `deep`, ... Passed through
    /// as is, since Exa adds modes over time.
    pub fn with_search_type(mut self, search_type: impl Into<String>) -> Self {
        self.search_type = Some(search_type.into());
        self
    }

    /// Focus on a data category: `news`, `company`, `publication`, `people`,
    /// `personal site`, `financial report`.
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Only these domains or domain paths (`example.com`, `example.com/docs`,
    /// `*.example.com`).
    pub fn with_include_domains<I, S>(mut self, domains: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.include_domains = domains.into_iter().map(Into::into).collect();
        self
    }

    /// Never these domains or domain paths.
    pub fn with_exclude_domains<I, S>(mut self, domains: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.exclude_domains = domains.into_iter().map(Into::into).collect();
        self
    }

    /// Only results published at or after this ISO 8601 date or date-time
    /// (`2026-01-01` or `2026-01-01T00:00:00Z`).
    pub fn with_start_published_date(mut self, date: &str) -> Result<Self, Box<dyn Error>> {
        self.start_published_date = Some(check_date(date)?);
        Ok(self)
    }

    /// Only results published at or before this ISO 8601 date or date-time.
    pub fn with_end_published_date(mut self, date: &str) -> Result<Self, Box<dyn Error>> {
        self.end_published_date = Some(check_date(date)?);
        Ok(self)
    }

    /// Page content to return with each result (highlights by default). A
    /// `max_characters` of `Some(0)` is rejected.
    pub fn with_contents(mut self, contents: ExaContents) -> Result<Self, Box<dyn Error>> {
        contents.check()?;
        self.contents = contents;
        Ok(self)
    }

    fn request_body(&self, query: &str) -> Value {
        let mut body = Map::new();
        body.insert("query".into(), json!(query));
        body.insert("numResults".into(), json!(self.num_results));
        let mut set = |key: &str, value: Option<Value>| {
            if let Some(value) = value {
                body.insert(key.into(), value);
            }
        };
        set("type", self.search_type.as_ref().map(|t| json!(t)));
        set("category", self.category.as_ref().map(|c| json!(c)));
        set(
            "includeDomains",
            (!self.include_domains.is_empty()).then(|| json!(self.include_domains)),
        );
        set(
            "excludeDomains",
            (!self.exclude_domains.is_empty()).then(|| json!(self.exclude_domains)),
        );
        set(
            "startPublishedDate",
            self.start_published_date.as_ref().map(|d| json!(d)),
        );
        set(
            "endPublishedDate",
            self.end_published_date.as_ref().map(|d| json!(d)),
        );
        set("contents", self.contents.to_json());
        Value::Object(body)
    }

    /// Runs one search.
    pub async fn search(&self, query: &str) -> Result<ExaSearchResults, Box<dyn Error>> {
        if query.trim().is_empty() {
            return Err("query cannot be empty".into());
        }
        if self.api_key.is_empty() {
            return Err("EXA_API_KEY is not set (or use with_api_key)".into());
        }
        let response = self
            .client
            .post(format!("{}/search", self.base_url))
            .header("x-api-key", &self.api_key)
            .json(&self.request_body(query))
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let text: String = text.chars().take(MAX_ERROR_BODY_CHARS).collect();
            return Err(format!("Exa search failed ({status}): {text}").into());
        }
        let response: SearchResponse = response.json().await?;
        for warning in &response.warnings {
            // Exa: "Log them; do not pass them to a model."
            tracing::warn!(%warning, "exa search warning");
        }
        Ok(ExaSearchResults {
            results: response.results,
            notices: response
                .notices
                .into_iter()
                .map(|n| n.message)
                .filter(|m| !m.is_empty())
                .collect(),
        })
    }
}

fn build_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("failed to build reqwest client")
}

/// `YYYY-MM-DD`, optionally followed by `T` and a time. Only the shape is checked; Exa
/// validates the rest.
fn check_date(date: &str) -> Result<String, Box<dyn Error>> {
    let bytes = date.as_bytes();
    let digits = |range: std::ops::Range<usize>| bytes[range].iter().all(u8::is_ascii_digit);
    let ok = bytes.len() >= 10
        && digits(0..4)
        && bytes[4] == b'-'
        && digits(5..7)
        && bytes[7] == b'-'
        && digits(8..10)
        && (bytes.len() == 10 || bytes[10] == b'T');
    if ok {
        Ok(date.to_string())
    } else {
        Err(format!("expected an ISO 8601 date like 2026-01-31, got {date:?}").into())
    }
}

fn format_results(search: &ExaSearchResults) -> String {
    let mut blocks: Vec<String> = search
        .results
        .iter()
        .enumerate()
        .map(|(i, result)| {
            let title = result
                .title
                .as_deref()
                .filter(|t| !t.is_empty())
                .unwrap_or("(untitled)");
            let mut text = format!("[{}] Title: {title}\nURL: {}", i + 1, result.url);
            if let Some(date) = result.published_date.as_deref().filter(|d| !d.is_empty()) {
                // Exa sends a full timestamp; the day is what matters to a reader.
                text.push_str(&format!("\nPublished: {}", date.get(..10).unwrap_or(date)));
            }
            if let Some(author) = result.author.as_deref().filter(|a| !a.is_empty()) {
                text.push_str(&format!("\nAuthor: {author}"));
            }
            let highlights: Vec<&str> = result
                .highlights
                .iter()
                .map(|h| h.trim())
                .filter(|h| !h.is_empty())
                .collect();
            if !highlights.is_empty() {
                text.push_str("\nHighlights:");
                for highlight in highlights {
                    text.push_str(&format!("\n- {highlight}"));
                }
            }
            if let Some(body) = result
                .text
                .as_deref()
                .map(str::trim)
                .filter(|t| !t.is_empty())
            {
                text.push_str(&format!("\nText: {body}"));
            }
            text
        })
        .collect();
    if !search.notices.is_empty() {
        blocks.push(format!(
            "Notes from the search engine:\n- {}",
            search.notices.join("\n- ")
        ));
    }
    blocks.join("\n\n")
}

#[async_trait]
impl Tool for ExaSearch {
    fn name(&self) -> String {
        "exa-search".to_string()
    }

    fn description(&self) -> String {
        "Searches the web with Exa and returns ranked results with title, URL, date and the \
         most relevant passages of each page. Useful for current events, research and facts \
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
        let search = self.search(query).await?;
        if search.results.is_empty() {
            let mut out = format!("No results found for query: {query}");
            if !search.notices.is_empty() {
                out.push_str(&format!(
                    "\n\nNotes from the search engine:\n- {}",
                    search.notices.join("\n- ")
                ));
            }
            return Ok(out);
        }
        Ok(format_results(&search))
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
