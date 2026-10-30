//! Page scraping through [Firecrawl](https://docs.firecrawl.dev/features/scrape).
//!
//! ```no_run
//! use langchain_rust::tools::{FirecrawlScrape, Tool};
//! use serde_json::json;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Reads FIRECRAWL_API_KEY.
//! let scrape = FirecrawlScrape::new()
//!     .with_exclude_tags(["nav", "footer"])
//!     .with_max_chars(8_000);
//! println!("{}", scrape.run(json!("https://www.rust-lang.org")).await?);
//! # Ok(())
//! # }
//! ```

use std::{error::Error, time::Duration};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::{build_client, input_field, non_empty, truncate, FirecrawlClient, FirecrawlMetadata};
use crate::tools::Tool;

/// A scraped page.
#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct FirecrawlDocument {
    #[serde(default)]
    pub markdown: Option<String>,
    #[serde(default)]
    pub metadata: FirecrawlMetadata,
}

#[derive(Debug, Deserialize)]
struct ScrapeData {
    #[serde(flatten)]
    document: FirecrawlDocument,
    #[serde(default)]
    warning: Option<String>,
}

/// Scrape tool backed by Firecrawl's `POST /v2/scrape`: one URL in, the page as markdown
/// out.
///
/// Firecrawl renders JavaScript, reads PDFs and strips navigation and boilerplate, so this
/// works on pages [`WebScrapper`](crate::tools::WebScrapper) gets little text from. Tool
/// output is `Title` and `URL` lines, the target page's status when it is not 2xx, then
/// the markdown. Use [`scrape`](Self::scrape) for the typed document.
#[derive(Debug, Clone)]
pub struct FirecrawlScrape {
    pub(super) client: FirecrawlClient,
    only_main_content: bool,
    include_tags: Vec<String>,
    exclude_tags: Vec<String>,
    wait_for: Option<Duration>,
    max_age: Option<Duration>,
    max_chars: Option<usize>,
}

impl Default for FirecrawlScrape {
    fn default() -> Self {
        Self::new()
    }
}

impl FirecrawlScrape {
    /// Reads the key from `FIRECRAWL_API_KEY` (calls fail with a clear error if it is
    /// unset; see [`with_api_key`](Self::with_api_key)).
    pub fn new() -> Self {
        Self {
            client: FirecrawlClient::new(),
            only_main_content: true,
            include_tags: Vec::new(),
            exclude_tags: Vec::new(),
            wait_for: None,
            max_age: None,
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

    /// Request timeout, 60 s by default.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client.http = build_client(timeout);
        self
    }

    /// Keep only the main content (the default) or return the whole page, headers and
    /// navigation included.
    pub fn with_only_main_content(mut self, only_main_content: bool) -> Self {
        self.only_main_content = only_main_content;
        self
    }

    /// Only these tags, classes or ids (`article`, `.content`, `#main`).
    pub fn with_include_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.include_tags = tags.into_iter().map(Into::into).collect();
        self
    }

    /// Drop these tags, classes or ids.
    pub fn with_exclude_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.exclude_tags = tags.into_iter().map(Into::into).collect();
        self
    }

    /// Wait this long after the page loads before reading it, for pages that render late.
    pub fn with_wait_for(mut self, wait_for: Duration) -> Self {
        self.wait_for = Some(wait_for);
        self
    }

    /// Accept a cached copy up to this old. `Duration::ZERO` always fetches a fresh copy.
    pub fn with_max_age(mut self, max_age: Duration) -> Self {
        self.max_age = Some(max_age);
        self
    }

    /// Cap the markdown in the tool output, so a long page doesn't fill the model's
    /// context. A cut ends with ` [truncated]`.
    pub fn with_max_chars(mut self, max_chars: usize) -> Self {
        self.max_chars = Some(max_chars);
        self
    }

    fn request_body(&self, url: &str) -> Map<String, Value> {
        let mut body = Map::new();
        body.insert("url".into(), json!(url));
        body.insert("formats".into(), json!(["markdown"]));
        body.insert("onlyMainContent".into(), json!(self.only_main_content));
        let mut set = |key: &str, value: Option<Value>| {
            if let Some(value) = value {
                body.insert(key.into(), value);
            }
        };
        set(
            "includeTags",
            (!self.include_tags.is_empty()).then(|| json!(self.include_tags)),
        );
        set(
            "excludeTags",
            (!self.exclude_tags.is_empty()).then(|| json!(self.exclude_tags)),
        );
        set(
            "waitFor",
            self.wait_for.map(|d| json!(d.as_millis() as u64)),
        );
        set("maxAge", self.max_age.map(|d| json!(d.as_millis() as u64)));
        body
    }

    /// Scrapes one http(s) URL.
    pub async fn scrape(&self, url: &str) -> Result<FirecrawlDocument, Box<dyn Error>> {
        let url = url.trim();
        let parsed = url::Url::parse(url).map_err(|e| format!("invalid URL {url:?}: {e}"))?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(format!("only http and https URLs can be scraped, got {url:?}").into());
        }
        let data: ScrapeData = self
            .client
            .post("/v2/scrape", self.request_body(url))
            .await?;
        if let Some(warning) = &data.warning {
            tracing::warn!(%warning, "firecrawl scrape warning");
        }
        Ok(data.document)
    }

    fn format_document(&self, requested: &str, document: &FirecrawlDocument) -> String {
        let metadata = &document.metadata;
        let url = non_empty(metadata.url.as_ref())
            .or_else(|| non_empty(metadata.source_url.as_ref()))
            .unwrap_or(requested);
        let mut text = String::new();
        if let Some(title) = non_empty(metadata.title.as_ref()) {
            text.push_str(&format!("Title: {title}\n"));
        }
        text.push_str(&format!("URL: {url}"));
        if let Some(status) = metadata.status_code.filter(|s| !(200..300).contains(s)) {
            text.push_str(&format!("\nStatus: {status}"));
        }
        if let Some(error) = non_empty(metadata.error.as_ref()) {
            text.push_str(&format!("\nError: {error}"));
        }
        match non_empty(document.markdown.as_ref()) {
            Some(markdown) => text.push_str(&format!("\n\n{}", truncate(markdown, self.max_chars))),
            None => text.push_str("\n\nNo content found on this page."),
        }
        text
    }
}

#[async_trait]
impl Tool for FirecrawlScrape {
    fn name(&self) -> String {
        "firecrawl-scrape".to_string()
    }

    fn description(&self) -> String {
        "Reads a web page with Firecrawl and returns its main content as markdown, with \
         JavaScript rendered and PDFs parsed. Useful for reading a page found by a search or \
         given by the user. Input should be a full URL."
            .to_string()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The http or https URL to read"
                }
            },
            "required": ["url"]
        })
    }

    /// Accepts a raw URL, `{"url": ...}` (tool calling) or `{"input": ...}`.
    async fn parse_input(&self, input: &str) -> Value {
        match serde_json::from_str::<Value>(input) {
            Ok(value @ Value::Object(_)) => value,
            _ => Value::String(input.to_string()),
        }
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let url = input_field(&input, "url")?.trim();
        let document = self.scrape(url).await?;
        Ok(self.format_document(url, &document))
    }
}
