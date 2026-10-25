//! Web search through the [Perplexity Search API](https://docs.perplexity.ai/docs/search/quickstart).
//!
//! The Search API returns ranked results (title, URL, snippet, dates), not a generated
//! answer, so the agent's own model reads the sources. langchaingo's Perplexity tool asks a
//! Sonar chat model instead; Perplexity now steers that use case to its Agent API, and a
//! chat model behind an OpenAI-compatible endpoint already works with
//! [`OpenAI`](crate::llm::openai::OpenAI) and `with_api_base("https://api.perplexity.ai")`.
//!
//! ```no_run
//! use langchain_rust::tools::{PerplexitySearch, SearchRecency, Tool};
//! use serde_json::json;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Reads PERPLEXITY_API_KEY.
//! let search = PerplexitySearch::new()
//!     .with_max_results(3)?
//!     .with_recency(SearchRecency::Week)
//!     .with_domains(["rust-lang.org", "github.com"])?;
//! println!("{}", search.run(json!("Rust 2024 edition")).await?);
//! # Ok(())
//! # }
//! ```

use std::{error::Error, time::Duration};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::tools::Tool;

pub const PERPLEXITY_API_BASE: &str = "https://api.perplexity.ai";
pub const DEFAULT_PERPLEXITY_MAX_RESULTS: u32 = 5;
const MAX_RESULTS_LIMIT: u32 = 20;
const MAX_DOMAINS: usize = 20;
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// Error bodies are quoted in the error, cut to this many characters.
const MAX_ERROR_BODY_CHARS: usize = 500;

/// `search_recency_filter`: only results published within the last hour, day, ...
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchRecency {
    Hour,
    Day,
    Week,
    Month,
    Year,
}

#[derive(Debug, Serialize)]
struct SearchRequest<'a> {
    query: &'a str,
    max_results: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens_per_page: Option<u32>,
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    search_domain_filter: &'a [String],
    #[serde(skip_serializing_if = "Option::is_none")]
    search_recency_filter: Option<SearchRecency>,
    #[serde(skip_serializing_if = "Option::is_none")]
    country: Option<&'a str>,
    #[serde(skip_serializing_if = "<[String]>::is_empty")]
    search_language_filter: &'a [String],
}

#[derive(Debug, Deserialize)]
struct SearchResponse {
    results: Vec<PerplexityResult>,
}

/// One ranked result from the Search API.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct PerplexityResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    #[serde(default)]
    pub date: Option<String>,
    #[serde(default)]
    pub last_updated: Option<String>,
}

/// Web search tool backed by Perplexity's `POST /search`.
///
/// The tool output lists each result as `Title`, `URL`, optional `Published` date and
/// `Snippet`, separated by blank lines. No results gives a plain message rather than an
/// error, so an agent can rephrase. Use [`search`](Self::search) for the typed results.
#[derive(Debug, Clone)]
pub struct PerplexitySearch {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    max_results: u32,
    max_tokens_per_page: Option<u32>,
    domains: Vec<String>,
    recency: Option<SearchRecency>,
    country: Option<String>,
    languages: Vec<String>,
}

impl Default for PerplexitySearch {
    fn default() -> Self {
        Self::new()
    }
}

impl PerplexitySearch {
    /// Reads the key from `PERPLEXITY_API_KEY` (calls fail with a clear error if it is
    /// unset; see [`with_api_key`](Self::with_api_key)).
    pub fn new() -> Self {
        Self {
            client: build_client(DEFAULT_TIMEOUT),
            base_url: PERPLEXITY_API_BASE.to_string(),
            api_key: std::env::var("PERPLEXITY_API_KEY").unwrap_or_default(),
            max_results: DEFAULT_PERPLEXITY_MAX_RESULTS,
            max_tokens_per_page: None,
            domains: Vec::new(),
            recency: None,
            country: None,
            languages: Vec::new(),
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// API origin, `https://api.perplexity.ai` by default. Mostly for tests and proxies.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Request timeout, 30 s by default.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = build_client(timeout);
        self
    }

    /// Results per query, 1 to 20 (default 5).
    pub fn with_max_results(mut self, max_results: u32) -> Result<Self, Box<dyn Error>> {
        if !(1..=MAX_RESULTS_LIMIT).contains(&max_results) {
            return Err(
                format!("max_results must be 1-{MAX_RESULTS_LIMIT}, got {max_results}").into(),
            );
        }
        self.max_results = max_results;
        Ok(self)
    }

    /// Caps the page content extracted into each snippet, in tokens. Snippets can be
    /// long, so set this when results go into a model's context.
    pub fn with_max_tokens_per_page(mut self, tokens: u32) -> Result<Self, Box<dyn Error>> {
        if tokens == 0 {
            return Err("max_tokens_per_page must be at least 1".into());
        }
        self.max_tokens_per_page = Some(tokens);
        Ok(self)
    }

    /// Only search these domains (up to 20), e.g. `["arxiv.org"]`.
    pub fn with_domains<I, S>(mut self, domains: I) -> Result<Self, Box<dyn Error>>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let domains: Vec<String> = domains.into_iter().map(Into::into).collect();
        if domains.len() > MAX_DOMAINS {
            return Err(format!("at most {MAX_DOMAINS} domains, got {}", domains.len()).into());
        }
        if let Some(bad) = domains.iter().find(|d| d.is_empty() || d.len() > 253) {
            return Err(format!("invalid domain {bad:?}").into());
        }
        self.domains = domains;
        Ok(self)
    }

    pub fn with_recency(mut self, recency: SearchRecency) -> Self {
        self.recency = Some(recency);
        self
    }

    /// ISO 3166-1 alpha-2 country code, e.g. `"PE"`.
    pub fn with_country(mut self, country: &str) -> Result<Self, Box<dyn Error>> {
        check_two_letter_code("country", country)?;
        self.country = Some(country.to_string());
        Ok(self)
    }

    /// ISO 639-1 language codes, e.g. `["es", "en"]` (up to 20).
    pub fn with_languages<I, S>(mut self, languages: I) -> Result<Self, Box<dyn Error>>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let languages: Vec<String> = languages.into_iter().map(Into::into).collect();
        if languages.len() > MAX_DOMAINS {
            return Err(format!("at most {MAX_DOMAINS} languages, got {}", languages.len()).into());
        }
        for language in &languages {
            check_two_letter_code("language", language)?;
        }
        self.languages = languages;
        Ok(self)
    }

    /// Runs one search and returns the ranked results.
    pub async fn search(&self, query: &str) -> Result<Vec<PerplexityResult>, Box<dyn Error>> {
        if query.trim().is_empty() {
            return Err("query cannot be empty".into());
        }
        if self.api_key.is_empty() {
            return Err("PERPLEXITY_API_KEY is not set (or use with_api_key)".into());
        }
        let body = SearchRequest {
            query,
            max_results: self.max_results,
            max_tokens_per_page: self.max_tokens_per_page,
            search_domain_filter: &self.domains,
            search_recency_filter: self.recency,
            country: self.country.as_deref(),
            search_language_filter: &self.languages,
        };
        let response = self
            .client
            .post(format!("{}/search", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let text: String = text.chars().take(MAX_ERROR_BODY_CHARS).collect();
            return Err(format!("Perplexity search failed ({status}): {text}").into());
        }
        Ok(response.json::<SearchResponse>().await?.results)
    }
}

fn build_client(timeout: Duration) -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .expect("failed to build reqwest client")
}

fn check_two_letter_code(what: &str, code: &str) -> Result<(), Box<dyn Error>> {
    if code.len() == 2 && code.chars().all(|c| c.is_ascii_alphabetic()) {
        Ok(())
    } else {
        Err(format!("{what} must be a two-letter code, got {code:?}").into())
    }
}

/// Renders results for a model to read.
fn format_results(results: &[PerplexityResult]) -> String {
    results
        .iter()
        .enumerate()
        .map(|(i, result)| {
            let mut text = format!("[{}] Title: {}\nURL: {}\n", i + 1, result.title, result.url);
            if let Some(date) = result.date.as_deref().filter(|d| !d.is_empty()) {
                text.push_str(&format!("Published: {date}\n"));
            }
            text.push_str(&format!("Snippet: {}", result.snippet.trim()));
            text
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[async_trait]
impl Tool for PerplexitySearch {
    fn name(&self) -> String {
        "perplexity-search".to_string()
    }

    fn description(&self) -> String {
        "Searches the web with Perplexity and returns ranked results with title, URL, \
         date and a content snippet. Useful for current events and facts that need a \
         source. Input should be a search query."
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
mod tests {
    use super::*;
    use mockito::Matcher;

    fn tool(server: &mockito::Server) -> PerplexitySearch {
        PerplexitySearch::new()
            .with_api_key("test-key")
            .with_base_url(server.url())
    }

    fn results_body() -> String {
        json!({
            "id": "abc",
            "results": [
                {
                    "title": "Rust 2024",
                    "url": "https://blog.rust-lang.org/2025/02/20/Rust-1.85.0.html",
                    "snippet": "  Rust 1.85 stabilizes the 2024 edition.\n",
                    "date": "2025-02-20",
                    "last_updated": "2025-03-01"
                },
                {
                    "title": "Edition guide",
                    "url": "https://doc.rust-lang.org/edition-guide/",
                    "snippet": "The edition guide.",
                    "date": null
                },
                {
                    "title": "Blank date",
                    "url": "https://example.com/",
                    "snippet": "x",
                    "date": ""
                }
            ]
        })
        .to_string()
    }

    #[tokio::test]
    async fn sends_query_and_defaults_with_bearer_auth() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/search")
            .match_header("authorization", "Bearer test-key")
            .match_header("content-type", "application/json")
            .match_body(Matcher::Json(
                json!({"query": "rust 2024", "max_results": 5}),
            ))
            .with_header("content-type", "application/json")
            .with_body(results_body())
            .create_async()
            .await;

        let out = tool(&server).run(json!("rust 2024")).await.unwrap();
        mock.assert_async().await;
        assert_eq!(
            out,
            "[1] Title: Rust 2024\n\
             URL: https://blog.rust-lang.org/2025/02/20/Rust-1.85.0.html\n\
             Published: 2025-02-20\n\
             Snippet: Rust 1.85 stabilizes the 2024 edition.\n\n\
             [2] Title: Edition guide\n\
             URL: https://doc.rust-lang.org/edition-guide/\n\
             Snippet: The edition guide.\n\n\
             [3] Title: Blank date\n\
             URL: https://example.com/\n\
             Snippet: x"
        );
    }

    #[tokio::test]
    async fn sends_every_filter() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/search")
            .match_body(Matcher::Json(json!({
                "query": "elecciones",
                "max_results": 2,
                "max_tokens_per_page": 256,
                "search_domain_filter": ["elcomercio.pe", "larepublica.pe"],
                "search_recency_filter": "week",
                "country": "PE",
                "search_language_filter": ["es"]
            })))
            .with_header("content-type", "application/json")
            .with_body(json!({"id": "x", "results": []}).to_string())
            .create_async()
            .await;

        let search = tool(&server)
            .with_max_results(2)
            .unwrap()
            .with_max_tokens_per_page(256)
            .unwrap()
            .with_domains(["elcomercio.pe", "larepublica.pe"])
            .unwrap()
            .with_recency(SearchRecency::Week)
            .with_country("PE")
            .unwrap()
            .with_languages(["es"])
            .unwrap();
        let out = search.run(json!("elecciones")).await.unwrap();
        mock.assert_async().await;
        assert_eq!(out, "No results found for query: elecciones");
    }

    #[tokio::test]
    async fn search_returns_typed_results() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/search")
            .with_header("content-type", "application/json")
            .with_body(results_body())
            .create_async()
            .await;

        let results = tool(&server).search("rust").await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].last_updated.as_deref(), Some("2025-03-01"));
        assert_eq!(results[1].date, None);
        assert_eq!(results[1].last_updated, None);
    }

    #[tokio::test]
    async fn accepts_tool_call_and_input_objects() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/search")
            .match_body(Matcher::PartialJson(json!({"query": "q1"})))
            .with_header("content-type", "application/json")
            .with_body(json!({"id": "x", "results": []}).to_string())
            .expect(3)
            .create_async()
            .await;

        let search = tool(&server);
        // What an agent passes to `call`: tool-call JSON, `{"input": ..}`, or plain text.
        for raw in [r#"{"query": "q1"}"#, r#"{"input": "q1"}"#, "q1"] {
            let out = search.call(raw).await.unwrap();
            assert_eq!(out, "No results found for query: q1", "{raw}");
        }
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn api_errors_carry_status_and_body() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/search")
            .with_status(401)
            .with_body(format!("{{\"error\": \"bad key\"}}{}", "x".repeat(2000)))
            .create_async()
            .await;

        let err = tool(&server).run(json!("q")).await.unwrap_err().to_string();
        assert!(err.contains("401"), "{err}");
        assert!(err.contains("bad key"), "{err}");
        assert!(err.len() < 600, "error body is cut: {} chars", err.len());
    }

    #[tokio::test]
    async fn malformed_response_is_an_error() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/search")
            .with_header("content-type", "application/json")
            .with_body(r#"{"answer": "no results field"}"#)
            .create_async()
            .await;
        assert!(tool(&server).run(json!("q")).await.is_err());
    }

    #[tokio::test]
    async fn times_out() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("POST", "/search")
            .with_chunked_body(|w| {
                std::thread::sleep(Duration::from_millis(1500));
                w.write_all(b"{}")
            })
            .create_async()
            .await;
        let search = tool(&server).with_timeout(Duration::from_millis(200));
        let started = std::time::Instant::now();
        let result = tokio::time::timeout(Duration::from_secs(5), search.run(json!("q")))
            .await
            .expect("client timeout should fire first");
        assert!(result.is_err());
        assert!(started.elapsed() < Duration::from_millis(1400));
    }

    #[tokio::test]
    async fn rejects_bad_input_before_any_request() {
        // Nothing listens on port 1; these must fail before sending.
        let search = PerplexitySearch::new()
            .with_api_key("k")
            .with_base_url("http://127.0.0.1:1");
        for input in [
            json!(""),
            json!("   "),
            json!(42),
            json!({"q": "x"}),
            json!({"query": 1}),
        ] {
            let err = search.run(input.clone()).await.unwrap_err().to_string();
            assert!(
                err.contains("empty") || err.contains("expected a string"),
                "{input}: {err}"
            );
        }
        let err = PerplexitySearch::new()
            .with_api_key("")
            .with_base_url("http://127.0.0.1:1")
            .run(json!("q"))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("PERPLEXITY_API_KEY"), "{err}");
    }

    #[test]
    fn option_validation() {
        let s = PerplexitySearch::new;
        assert!(s().with_max_results(0).is_err());
        assert!(s().with_max_results(21).is_err());
        assert_eq!(s().with_max_results(1).unwrap().max_results, 1);
        assert_eq!(s().with_max_results(20).unwrap().max_results, 20);
        assert!(s().with_max_tokens_per_page(0).is_err());
        assert!(s().with_domains(vec!["a.com"; 21]).is_err());
        assert_eq!(
            s().with_domains(vec!["a.com"; 20]).unwrap().domains.len(),
            20
        );
        assert!(s().with_domains([""]).is_err());
        assert!(s().with_domains(["a".repeat(254)]).is_err());
        assert!(s().with_domains(["a".repeat(253)]).is_ok());
        for bad in ["", "P", "PER", "P1", "é"] {
            assert!(s().with_country(bad).is_err(), "{bad}");
            assert!(s().with_languages([bad]).is_err(), "{bad}");
        }
        assert!(s().with_languages(vec!["es"; 21]).is_err());
        assert_eq!(
            s().with_languages(vec!["es"; 20]).unwrap().languages.len(),
            20
        );
        assert_eq!(
            s().with_base_url("http://x/").base_url,
            "http://x",
            "trailing slash trimmed"
        );
    }

    #[tokio::test]
    async fn agent_reads_search_results() {
        use crate::{
            agent::{AgentExecutor, OpenAiToolAgentBuilder},
            chain::Chain,
            prompt_args,
            schemas::{FunctionCallResponse, FunctionDetail, MessageType},
            test_utils::FakeLLM,
        };
        use std::sync::Arc;

        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/search")
            .match_body(Matcher::PartialJson(json!({"query": "rust 2024 edition"})))
            .with_header("content-type", "application/json")
            .with_body(results_body())
            .create_async()
            .await;

        let call = FunctionCallResponse {
            id: "call_1".into(),
            type_field: "function".into(),
            function: FunctionDetail {
                name: "perplexity-search".into(),
                arguments: r#"{"query": "rust 2024 edition"}"#.into(),
            },
        };
        let llm = FakeLLM::new([
            serde_json::to_string(&vec![call]).unwrap(),
            "Rust 1.85 shipped it.".to_string(),
        ]);
        let agent = OpenAiToolAgentBuilder::new()
            .tools(&[Arc::new(tool(&server))])
            .build(llm.clone())
            .unwrap();
        let out = AgentExecutor::from_agent(agent)
            .invoke(prompt_args! {"input" => "when did the 2024 edition ship?"})
            .await
            .unwrap();
        assert_eq!(out, "Rust 1.85 shipped it.");
        mock.assert_async().await;

        let calls = llm.calls();
        let observation = calls[1]
            .iter()
            .find(|m| m.message_type == MessageType::ToolMessage)
            .expect("search results go back to the model");
        assert!(observation.content.starts_with("[1] Title: Rust 2024\n"));
        assert!(observation
            .content
            .contains("URL: https://doc.rust-lang.org/edition-guide/"));
    }

    /// A real model against a mocked Search API: picks the tool, sends `{"query": ..}`,
    /// and answers from the results.
    /// `OLLAMA_MODEL=qwen3:14b cargo test --lib perplexity_agent_live -- --ignored --nocapture`
    #[tokio::test]
    #[ignore]
    async fn perplexity_agent_live_ollama() {
        use crate::{
            agent::{AgentExecutor, OpenAiToolAgentBuilder},
            chain::Chain,
            llm::{ollama::openai::OllamaConfig, openai::OpenAI},
            prompt_args,
        };
        use std::sync::Arc;

        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/search")
            .match_body(Matcher::Regex("(?i)zorblax".into()))
            .with_header("content-type", "application/json")
            .with_body(
                json!({"id": "x", "results": [{
                    "title": "Zorblax 9 launch",
                    "url": "https://example.com/zorblax",
                    "snippet": "The Zorblax 9 phone launched on 3 March 2031 in Lima.",
                    "date": "2031-03-03"
                }]})
                .to_string(),
            )
            .expect_at_least(1)
            .create_async()
            .await;

        let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:14b".into());
        let agent = OpenAiToolAgentBuilder::new()
            .tools(&[Arc::new(tool(&server))])
            .build(OpenAI::new(OllamaConfig::default()).with_model(&model))
            .unwrap();
        let out = AgentExecutor::from_agent(agent)
            .invoke(prompt_args! {
                "input" => "Search the web: in which city did the Zorblax 9 phone launch? Answer in one sentence."
            })
            .await
            .unwrap();
        println!("{out}");
        mock.assert_async().await;
        assert!(out.contains("Lima"), "{out}");
    }

    #[test]
    fn tool_metadata() {
        let search = PerplexitySearch::new();
        assert_eq!(search.name(), "perplexity-search");
        assert_eq!(search.parameters()["required"], json!(["query"]));
        assert!(search.description().contains("search query"));
    }

    /// `PERPLEXITY_API_KEY=... cargo test --lib perplexity_live -- --ignored`
    #[tokio::test]
    #[ignore]
    async fn perplexity_live() {
        if std::env::var("PERPLEXITY_API_KEY").is_err() {
            eprintln!("PERPLEXITY_API_KEY not set, skipping");
            return;
        }
        let search = PerplexitySearch::new()
            .with_max_results(3)
            .unwrap()
            .with_max_tokens_per_page(128)
            .unwrap();
        let results = search.search("Rust programming language").await.unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|r| r.url.starts_with("http")));
    }
}
