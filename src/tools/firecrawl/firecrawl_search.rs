//! Web search through [Firecrawl](https://docs.firecrawl.dev/features/search).
//!
//! ```no_run
//! use langchain_rust::tools::{FirecrawlSearch, Tool};
//! use serde_json::json;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Reads FIRECRAWL_API_KEY.
//! let search = FirecrawlSearch::new()
//!     .with_limit(3)?
//!     .with_tbs("qdr:w")
//!     .with_scrape_results(true)
//!     .with_max_chars(2_000);
//! println!("{}", search.run(json!("rust async runtime benchmarks")).await?);
//! # Ok(())
//! # }
//! ```

use std::{error::Error, time::Duration};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::{build_client, input_field, non_empty, truncate, FirecrawlClient, FirecrawlMetadata};
use crate::tools::Tool;

pub const DEFAULT_FIRECRAWL_SEARCH_LIMIT: u32 = 5;
const MAX_LIMIT: u32 = 100;

/// One web result. `markdown` and `metadata` are set when results are scraped.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FirecrawlSearchResult {
    pub url: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub markdown: Option<String>,
    #[serde(default)]
    pub metadata: Option<FirecrawlMetadata>,
}

#[derive(Debug, Deserialize)]
struct SearchData {
    #[serde(default)]
    web: Vec<FirecrawlSearchResult>,
}

/// Web search tool backed by Firecrawl's `POST /v2/search`.
///
/// Tool output: numbered `Title / URL / Description` blocks, plus each page's markdown
/// when [`with_scrape_results`](Self::with_scrape_results) is on. No results gives a plain
/// message instead of an error. Use [`search`](Self::search) for typed results.
#[derive(Debug, Clone)]
pub struct FirecrawlSearch {
    pub(super) client: FirecrawlClient,
    pub(super) limit: u32,
    tbs: Option<String>,
    location: Option<String>,
    country: Option<String>,
    include_domains: Vec<String>,
    exclude_domains: Vec<String>,
    scrape_results: bool,
    max_chars: Option<usize>,
}

impl Default for FirecrawlSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl FirecrawlSearch {
    /// Reads the key from `FIRECRAWL_API_KEY` (calls fail with a clear error if it is
    /// unset; see [`with_api_key`](Self::with_api_key)).
    pub fn new() -> Self {
        Self {
            client: FirecrawlClient::new(),
            limit: DEFAULT_FIRECRAWL_SEARCH_LIMIT,
            tbs: None,
            location: None,
            country: None,
            include_domains: Vec::new(),
            exclude_domains: Vec::new(),
            scrape_results: false,
            max_chars: None,
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.client.api_key = api_key.into();
        self
    }

    /// API origin, `https://api.firecrawl.dev` by default. Point it at a self-hosted
    /// Firecrawl, a proxy or a test server.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.client.set_base_url(base_url.into());
        self
    }

    /// Request timeout, 60 s by default (scraping results takes longer than a plain search).
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client.http = build_client(timeout);
        self
    }

    /// Results per query, 1 to 100 (default 5).
    pub fn with_limit(mut self, limit: u32) -> Result<Self, Box<dyn Error>> {
        if !(1..=MAX_LIMIT).contains(&limit) {
            return Err(format!("limit must be 1-{MAX_LIMIT}, got {limit}").into());
        }
        self.limit = limit;
        Ok(self)
    }

    /// Time filter, passed through: `qdr:h`, `qdr:d`, `qdr:w`, `qdr:m`, `qdr:y` or a
    /// custom range like `cdr:1,cd_min:1/1/2026,cd_max:1/31/2026`.
    pub fn with_tbs(mut self, tbs: impl Into<String>) -> Self {
        self.tbs = Some(tbs.into());
        self
    }

    /// Search from a place, like `Germany` or `San Francisco,California,United States`.
    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    /// ISO country code for results, like `US` or `DE`.
    pub fn with_country(mut self, country: impl Into<String>) -> Self {
        self.country = Some(country.into());
        self
    }

    /// Only these domains. Can't be combined with
    /// [`with_exclude_domains`](Self::with_exclude_domains).
    pub fn with_include_domains<I, S>(mut self, domains: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.include_domains = domains.into_iter().map(Into::into).collect();
        self
    }

    /// Never these domains.
    pub fn with_exclude_domains<I, S>(mut self, domains: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.exclude_domains = domains.into_iter().map(Into::into).collect();
        self
    }

    /// Also scrape every result and return its main content as markdown (off by default).
    /// Each scraped page costs credits on top of the search.
    pub fn with_scrape_results(mut self, scrape_results: bool) -> Self {
        self.scrape_results = scrape_results;
        self
    }

    /// Cap each result's markdown in the tool output, so a few long pages don't fill the
    /// model's context. A cut ends with ` [truncated]`.
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = Some(max_chars);
        self
    }

    fn request_body(&self, query: &str) -> Map<String, Value> {
        let mut body = Map::new();
        body.insert("query".into(), json!(query));
        body.insert("limit".into(), json!(self.limit));
        let mut set = |key: &str, value: Option<Value>| {
            if let Some(value) = value {
                body.insert(key.into(), value);
            }
        };
        set("tbs", self.tbs.as_ref().map(|t| json!(t)));
        set("location", self.location.as_ref().map(|l| json!(l)));
        set("country", self.country.as_ref().map(|c| json!(c)));
        set(
            "includeDomains",
            (!self.include_domains.is_empty()).then(|| json!(self.include_domains)),
        );
        set(
            "excludeDomains",
            (!self.exclude_domains.is_empty()).then(|| json!(self.exclude_domains)),
        );
        set(
            "scrapeOptions",
            self.scrape_results
                .then(|| json!({"formats": ["markdown"], "onlyMainContent": true})),
        );
        body
    }

    /// Runs one search and returns the web results.
    pub async fn search(&self, query: &str) -> Result<Vec<FirecrawlSearchResult>, Box<dyn Error>> {
        if query.trim().is_empty() {
            return Err("query cannot be empty".into());
        }
        if !self.include_domains.is_empty() && !self.exclude_domains.is_empty() {
            return Err("include_domains and exclude_domains cannot both be set".into());
        }
        let data: SearchData = self
            .client
            .post("/v2/search", self.request_body(query))
            .await?;
        Ok(data.web)
    }

    fn format_results(&self, results: &[FirecrawlSearchResult]) -> String {
        results
            .iter()
            .enumerate()
            .map(|(i, result)| {
                let metadata = result.metadata.as_ref();
                let title = non_empty(result.title.as_ref())
                    .or_else(|| non_empty(metadata.and_then(|m| m.title.as_ref())))
                    .unwrap_or("(untitled)");
                let mut text = format!("[{}] Title: {title}\nURL: {}", i + 1, result.url);
                if let Some(description) = non_empty(result.description.as_ref())
                    .or_else(|| non_empty(metadata.and_then(|m| m.description.as_ref())))
                {
                    text.push_str(&format!("\nDescription: {description}"));
                }
                if let Some(markdown) = non_empty(result.markdown.as_ref()) {
                    text.push_str(&format!(
                        "\nContent:\n{}",
                        truncate(markdown, self.max_chars)
                    ));
                }
                text
            })
            .collect::<Vec<_>>()
            .join("\n\n")
    }
}

#[async_trait]
impl Tool for FirecrawlSearch {
    fn name(&self) -> String {
        "firecrawl-search".to_string()
    }

    fn description(&self) -> String {
        "Searches the web with Firecrawl and returns ranked results with title, URL and a \
         short description, plus page content when enabled. Useful for current events, \
         research and facts that need a source. Input should be a search query."
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
        let query = input_field(&input, "query")?;
        let results = self.search(query).await?;
        if results.is_empty() {
            return Ok(format!("No results found for query: {query}"));
        }
        Ok(self.format_results(&results))
    }
}
