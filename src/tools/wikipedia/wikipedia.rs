//! Wikipedia API Tool for LangChain Rust
//!
//! This module provides a tool for searching and fetching content from Wikipedia
//! using the MediaWiki API. It implements the `Tool` trait to be used with
//! LangChain agents and chains.
//!
//! # Examples
//!
//! ```no_run
//! use langchain_rust::tools::{Tool, WikipediaQuery};
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() {
//!     let wiki = WikipediaQuery::default();
//!     let result = wiki.run(json!("Rust programming language")).await.unwrap();
//!     println!("{}", result);
//! }
//! ```
//!
//! # Configuration
//!
//! Use [`WikipediaQueryOptions`] or the builder methods on [`WikipediaQuery`] to
//! control how many results are fetched, how long each summary may be, and which
//! language edition of Wikipedia is queried:
//!
//! ```no_run
//! use langchain_rust::tools::WikipediaQuery;
//!
//! let wiki = WikipediaQuery::with_lang("es")
//!     .with_top_k_results(2)
//!     .with_max_doc_content_length(1000);
//! ```
//!
//! # Output format
//!
//! Each matching article is rendered as a `Page: <title>` line followed by a
//! `Summary: <intro extract>` line. Multiple articles are separated by a blank
//! line. If nothing matches, a `No results found for query: ...` message is
//! returned rather than an error, so agents can continue reasoning.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::error::Error;

use crate::tools::Tool;

/// User agent sent with every request. Wikipedia rejects requests without one.
const APP_USER_AGENT: &str = concat!(
    env!("CARGO_PKG_NAME"),
    "/",
    env!("CARGO_PKG_VERSION"),
    " https://github.com/Abraxas-365/langchain-rust"
);

/// Configuration options for Wikipedia queries
#[derive(Debug, Clone)]
pub struct WikipediaQueryOptions {
    /// Maximum number of search results to return
    pub top_k_results: usize,
    /// Maximum length of each document content in characters
    pub max_doc_content_length: usize,
    /// Language code for Wikipedia (e.g., "en", "es", "fr").
    ///
    /// Must match `[a-z-]+`, since it is used to build the API hostname.
    pub lang: String,
}

impl Default for WikipediaQueryOptions {
    fn default() -> Self {
        Self {
            top_k_results: 3,
            max_doc_content_length: 4000,
            lang: "en".to_string(),
        }
    }
}

/// Wikipedia API response structures
#[derive(Debug, Deserialize)]
struct WikipediaSearchResponse {
    query: SearchQuery,
}

#[derive(Debug, Deserialize)]
struct SearchQuery {
    search: Vec<SearchResult>,
}

#[derive(Debug, Deserialize)]
struct SearchResult {
    title: String,
}

#[derive(Debug, Deserialize)]
struct WikipediaPageResponse {
    query: PageQuery,
}

#[derive(Debug, Deserialize)]
struct PageQuery {
    pages: std::collections::HashMap<String, PageContent>,
}

#[derive(Debug, Deserialize)]
struct PageContent {
    title: String,
    extract: Option<String>,
}

/// A tool for querying Wikipedia articles
///
/// This tool searches Wikipedia and returns summaries of the top matching articles.
/// It's useful for answering general questions about people, places, companies, facts,
/// historical events, and other subjects that have Wikipedia articles.
///
/// # Examples
///
/// ```no_run
/// use langchain_rust::tools::{Tool, WikipediaQuery, WikipediaQueryOptions};
/// use serde_json::json;
///
/// #[tokio::main]
/// async fn main() {
///     let options = WikipediaQueryOptions {
///         top_k_results: 2,
///         max_doc_content_length: 2000,
///         lang: "en".to_string(),
///     };
///
///     let wiki = WikipediaQuery::new(options);
///     let result = wiki.run(json!("LangChain")).await.unwrap();
///     println!("{}", result);
/// }
/// ```
pub struct WikipediaQuery {
    pub options: WikipediaQueryOptions,
    client: reqwest::Client,
    /// Overrides the `https://{lang}.wikipedia.org` origin. Used for tests.
    base_url: Option<String>,
}

impl WikipediaQuery {
    /// Creates a new WikipediaQuery with custom options
    ///
    /// # Arguments
    ///
    /// * `options` - Configuration options for Wikipedia queries
    pub fn new(options: WikipediaQueryOptions) -> Self {
        let client = reqwest::Client::builder()
            .user_agent(APP_USER_AGENT)
            .build()
            .expect("failed to build reqwest client");

        Self {
            options,
            client,
            base_url: None,
        }
    }

    /// Creates a WikipediaQuery with a specific language
    ///
    /// # Arguments
    ///
    /// * `lang` - Language code (e.g., "en", "es", "fr", "de")
    pub fn with_lang(lang: impl Into<String>) -> Self {
        Self::new(WikipediaQueryOptions {
            lang: lang.into(),
            ..Default::default()
        })
    }

    /// Sets the maximum number of results to return
    pub fn with_top_k_results(mut self, top_k: usize) -> Self {
        self.options.top_k_results = top_k;
        self
    }

    /// Sets the maximum content length per document
    pub fn with_max_doc_content_length(mut self, max_len: usize) -> Self {
        self.options.max_doc_content_length = max_len;
        self
    }

    /// Overrides the API origin (e.g. `http://127.0.0.1:1234`) instead of
    /// deriving it from the configured language. Mainly useful for pointing the
    /// tool at a mock server in tests or at a self-hosted MediaWiki instance.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Builds the Wikipedia API URL for the configured language
    fn get_api_url(&self) -> Result<String, Box<dyn Error>> {
        let origin = match &self.base_url {
            Some(base) => base.trim_end_matches('/').to_string(),
            None => {
                let lang = &self.options.lang;
                let valid =
                    !lang.is_empty() && lang.chars().all(|c| c.is_ascii_lowercase() || c == '-');
                if !valid {
                    return Err(format!(
                        "Invalid Wikipedia language code '{}': must match [a-z-]+",
                        lang
                    )
                    .into());
                }
                format!("https://{}.wikipedia.org", lang)
            }
        };
        Ok(format!("{}/w/api.php", origin))
    }

    /// Searches Wikipedia for articles matching the query
    async fn search(&self, query: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let api_url = self.get_api_url()?;
        let params = [
            ("action", "query"),
            ("list", "search"),
            ("srsearch", query),
            ("format", "json"),
            ("srlimit", &self.options.top_k_results.to_string()),
        ];

        let response = self
            .client
            .get(&api_url)
            .query(&params)
            .send()
            .await?
            .error_for_status()?
            .json::<WikipediaSearchResponse>()
            .await?;

        Ok(response.query.search.into_iter().map(|r| r.title).collect())
    }

    /// Fetches the content of a specific Wikipedia page
    async fn fetch_page(&self, title: &str) -> Result<String, Box<dyn Error>> {
        let api_url = self.get_api_url()?;
        let params = [
            ("action", "query"),
            ("prop", "extracts"),
            ("titles", title),
            ("format", "json"),
            ("explaintext", "true"),
            ("exintro", "true"),
        ];

        let response = self
            .client
            .get(&api_url)
            .query(&params)
            .send()
            .await?
            .error_for_status()?
            .json::<WikipediaPageResponse>()
            .await?;

        match response.query.pages.values().next() {
            Some(page) => {
                let extract = page.extract.as_deref().unwrap_or("");
                let truncated: String = extract
                    .chars()
                    .take(self.options.max_doc_content_length)
                    .collect();
                Ok(format!("Page: {}\nSummary: {}", page.title, truncated))
            }
            None => Err("Page not found".into()),
        }
    }
}

impl Default for WikipediaQuery {
    fn default() -> Self {
        Self::new(WikipediaQueryOptions::default())
    }
}

#[async_trait]
impl Tool for WikipediaQuery {
    fn name(&self) -> String {
        "wikipedia-api".to_string()
    }

    fn description(&self) -> String {
        "A wrapper around Wikipedia. \
         Useful for when you need to answer general questions about \
         people, places, companies, facts, historical events, or other subjects. \
         Input should be a search query."
            .to_string()
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let query = match input {
            Value::String(s) => s,
            Value::Object(mut map) => map
                .remove("input")
                .and_then(|v| v.as_str().map(String::from))
                .ok_or("Invalid input format: expected 'input' field")?,
            _ => return Err("Input must be a string or object with 'input' field".into()),
        };

        if query.trim().is_empty() {
            return Err("Query cannot be empty".into());
        }

        // Search for relevant pages
        let titles = self.search(&query).await?;

        if titles.is_empty() {
            return Ok(format!("No results found for query: {}", query));
        }

        // Fetch content for all found pages, skipping any that fail individually
        let mut results = Vec::new();
        for title in &titles {
            if let Ok(content) = self.fetch_page(title).await {
                results.push(content);
            }
        }

        if results.is_empty() {
            return Ok("No page content could be retrieved".to_string());
        }

        Ok(results.join("\n\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::Matcher;
    use serde_json::json;

    #[test]
    fn test_default_options() {
        let options = WikipediaQueryOptions::default();
        assert_eq!(options.top_k_results, 3);
        assert_eq!(options.max_doc_content_length, 4000);
        assert_eq!(options.lang, "en");
    }

    #[test]
    fn test_custom_options() {
        let options = WikipediaQueryOptions {
            top_k_results: 5,
            max_doc_content_length: 2000,
            lang: "es".to_string(),
        };
        assert_eq!(options.top_k_results, 5);
        assert_eq!(options.max_doc_content_length, 2000);
        assert_eq!(options.lang, "es");
    }

    #[test]
    fn test_tool_name() {
        let wiki = WikipediaQuery::default();
        assert_eq!(wiki.name(), "wikipedia-api");
    }

    #[test]
    fn test_tool_description() {
        let wiki = WikipediaQuery::default();
        let desc = wiki.description();
        assert!(desc.contains("Wikipedia"));
        assert!(desc.contains("search query"));
    }

    #[test]
    fn test_builder_pattern() {
        let wiki = WikipediaQuery::default()
            .with_top_k_results(5)
            .with_max_doc_content_length(2000);

        assert_eq!(wiki.options.top_k_results, 5);
        assert_eq!(wiki.options.max_doc_content_length, 2000);
    }

    #[test]
    fn test_with_lang() {
        let wiki = WikipediaQuery::with_lang("fr");
        assert_eq!(wiki.options.lang, "fr");
        assert_eq!(
            wiki.get_api_url().unwrap(),
            "https://fr.wikipedia.org/w/api.php"
        );
    }

    #[test]
    fn test_api_url_construction() {
        let wiki_en = WikipediaQuery::default();
        assert_eq!(
            wiki_en.get_api_url().unwrap(),
            "https://en.wikipedia.org/w/api.php"
        );

        let wiki_es = WikipediaQuery::with_lang("es");
        assert_eq!(
            wiki_es.get_api_url().unwrap(),
            "https://es.wikipedia.org/w/api.php"
        );

        let wiki_hyphen = WikipediaQuery::with_lang("zh-yue");
        assert_eq!(
            wiki_hyphen.get_api_url().unwrap(),
            "https://zh-yue.wikipedia.org/w/api.php"
        );
    }

    #[test]
    fn test_invalid_lang_rejected() {
        for bad in ["", "EN", "en.evil.com", "en/", "e n", "ja_JP"] {
            let wiki = WikipediaQuery::with_lang(bad);
            let err = wiki.get_api_url().unwrap_err().to_string();
            assert!(err.contains("Invalid Wikipedia language code"), "{bad:?}");
        }
    }

    #[test]
    fn test_base_url_override() {
        let wiki = WikipediaQuery::default().with_base_url("http://127.0.0.1:9999/");
        assert_eq!(
            wiki.get_api_url().unwrap(),
            "http://127.0.0.1:9999/w/api.php"
        );
    }

    #[tokio::test]
    async fn test_empty_query_error() {
        let wiki = WikipediaQuery::default();
        let result = wiki.run(json!("")).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[tokio::test]
    async fn test_invalid_input_format() {
        let wiki = WikipediaQuery::default();
        let result = wiki.run(json!(123)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_invalid_lang_errors_at_run() {
        let wiki = WikipediaQuery::with_lang("en.evil.com");
        let err = wiki.run(json!("anything")).await.unwrap_err().to_string();
        assert!(err.contains("Invalid Wikipedia language code"));
    }

    // ---------------- mock server tests ----------------

    fn search_body(titles: &[&str]) -> String {
        json!({
            "query": {
                "search": titles.iter().map(|t| json!({"title": t})).collect::<Vec<_>>()
            }
        })
        .to_string()
    }

    fn page_body(title: &str, extract: &str) -> String {
        json!({
            "query": {
                "pages": {
                    "1": {"title": title, "extract": extract}
                }
            }
        })
        .to_string()
    }

    fn mock_search(
        server: &mut mockito::ServerGuard,
        query: &str,
        titles: &[&str],
    ) -> mockito::Mock {
        server
            .mock("GET", "/w/api.php")
            .match_query(Matcher::AllOf(vec![
                Matcher::UrlEncoded("action".into(), "query".into()),
                Matcher::UrlEncoded("list".into(), "search".into()),
                Matcher::UrlEncoded("srsearch".into(), query.into()),
            ]))
            .with_header("content-type", "application/json")
            .with_body(search_body(titles))
            .create()
    }

    fn mock_page(server: &mut mockito::ServerGuard, title: &str, extract: &str) -> mockito::Mock {
        server
            .mock("GET", "/w/api.php")
            .match_query(Matcher::AllOf(vec![
                Matcher::UrlEncoded("prop".into(), "extracts".into()),
                Matcher::UrlEncoded("titles".into(), title.into()),
            ]))
            .with_header("content-type", "application/json")
            .with_body(page_body(title, extract))
            .create()
    }

    #[tokio::test]
    async fn test_run_against_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let search = mock_search(
            &mut server,
            "rust",
            &["Rust (programming language)", "Rust"],
        );
        let page1 = mock_page(
            &mut server,
            "Rust (programming language)",
            "Rust is a systems language.",
        );
        let page2 = mock_page(&mut server, "Rust", "Rust is an iron oxide.");

        let wiki = WikipediaQuery::default().with_base_url(server.url());
        let out = wiki.run(json!("rust")).await.unwrap();

        search.assert();
        page1.assert();
        page2.assert();
        assert_eq!(
            out,
            "Page: Rust (programming language)\nSummary: Rust is a systems language.\n\n\
             Page: Rust\nSummary: Rust is an iron oxide."
        );
    }

    #[tokio::test]
    async fn test_object_input_against_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let _search = mock_search(&mut server, "ferris", &["Ferris"]);
        let _page = mock_page(&mut server, "Ferris", "Ferris is a crab.");

        let wiki = WikipediaQuery::default().with_base_url(server.url());
        let out = wiki.run(json!({"input": "ferris"})).await.unwrap();
        assert_eq!(out, "Page: Ferris\nSummary: Ferris is a crab.");
    }

    #[tokio::test]
    async fn test_no_results_from_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let _search = mock_search(&mut server, "nothing", &[]);

        let wiki = WikipediaQuery::default().with_base_url(server.url());
        let out = wiki.run(json!("nothing")).await.unwrap();
        assert_eq!(out, "No results found for query: nothing");
    }

    #[tokio::test]
    async fn test_truncation_counts_chars_not_bytes() {
        let mut server = mockito::Server::new_async().await;
        let _search = mock_search(&mut server, "日本", &["日本"]);
        // Multi-byte characters: byte slicing at 5 would panic or split a char.
        let _page = mock_page(&mut server, "日本", "日本は島国です。");

        let wiki = WikipediaQuery::default()
            .with_base_url(server.url())
            .with_max_doc_content_length(5);
        let out = wiki.run(json!("日本")).await.unwrap();
        assert_eq!(out, "Page: 日本\nSummary: 日本は島国");
    }

    #[tokio::test]
    async fn test_zero_max_length_does_not_panic() {
        let mut server = mockito::Server::new_async().await;
        let _search = mock_search(&mut server, "x", &["X"]);
        let _page = mock_page(&mut server, "X", "some text");

        let wiki = WikipediaQuery::default()
            .with_base_url(server.url())
            .with_max_doc_content_length(0);
        let out = wiki.run(json!("x")).await.unwrap();
        assert_eq!(out, "Page: X\nSummary: ");
    }

    #[tokio::test]
    async fn test_search_http_error_is_propagated() {
        let mut server = mockito::Server::new_async().await;
        let _search = server
            .mock("GET", "/w/api.php")
            .match_query(Matcher::UrlEncoded("list".into(), "search".into()))
            .with_status(503)
            .with_body("<html>Service Unavailable</html>")
            .create();

        let wiki = WikipediaQuery::default().with_base_url(server.url());
        let err = wiki.run(json!("rust")).await.unwrap_err().to_string();
        assert!(err.contains("503"), "unexpected error: {err}");
    }

    #[tokio::test]
    async fn test_failed_page_fetch_is_skipped() {
        let mut server = mockito::Server::new_async().await;
        let _search = mock_search(&mut server, "mixed", &["Good", "Bad"]);
        let _good = mock_page(&mut server, "Good", "fine");
        let _bad = server
            .mock("GET", "/w/api.php")
            .match_query(Matcher::AllOf(vec![
                Matcher::UrlEncoded("prop".into(), "extracts".into()),
                Matcher::UrlEncoded("titles".into(), "Bad".into()),
            ]))
            .with_status(500)
            .create();

        let wiki = WikipediaQuery::default().with_base_url(server.url());
        let out = wiki.run(json!("mixed")).await.unwrap();
        assert_eq!(out, "Page: Good\nSummary: fine");
    }

    #[tokio::test]
    async fn test_all_page_fetches_failing_returns_message() {
        let mut server = mockito::Server::new_async().await;
        let _search = mock_search(&mut server, "broken", &["Bad"]);
        let _bad = server
            .mock("GET", "/w/api.php")
            .match_query(Matcher::UrlEncoded("prop".into(), "extracts".into()))
            .with_status(500)
            .create();

        let wiki = WikipediaQuery::default().with_base_url(server.url());
        let out = wiki.run(json!("broken")).await.unwrap();
        assert_eq!(out, "No page content could be retrieved");
    }

    // Integration tests - these require internet connection
    #[tokio::test]
    #[ignore] // Remove ignore to run with network access
    async fn test_search_integration() {
        let wiki = WikipediaQuery::default();
        let titles = wiki.search("Rust programming language").await.unwrap();
        assert!(!titles.is_empty());
        assert!(titles.len() <= 3); // Default top_k_results
    }

    #[tokio::test]
    #[ignore] // Remove ignore to run with network access
    async fn test_fetch_page_integration() {
        let wiki = WikipediaQuery::default();
        let content = wiki
            .fetch_page("Rust (programming language)")
            .await
            .unwrap();
        assert!(content.contains("Page:"));
        assert!(content.contains("Summary:"));
    }

    #[tokio::test]
    #[ignore] // Remove ignore to run with network access
    async fn test_run_integration() {
        let wiki = WikipediaQuery::default();
        let result = wiki.run(json!("Rust programming language")).await.unwrap();
        assert!(result.contains("Page:"));
        assert!(result.contains("Summary:"));
    }

    #[tokio::test]
    #[ignore] // Remove ignore to run with network access
    async fn test_spanish_wikipedia() {
        let wiki = WikipediaQuery::with_lang("es");
        let result = wiki.run(json!("Rust lenguaje programación")).await.unwrap();
        assert!(result.contains("Page:"));
    }

    #[tokio::test]
    #[ignore] // Remove ignore to run with network access
    async fn test_content_truncation() {
        let wiki = WikipediaQuery::default().with_max_doc_content_length(100);
        let result = wiki.run(json!("United States")).await.unwrap();
        // The summary for each page should be truncated
        for line in result.lines() {
            if let Some(summary) = line.strip_prefix("Summary: ") {
                assert!(summary.chars().count() <= 100);
            }
        }
    }

    #[tokio::test]
    #[ignore] // Remove ignore to run with network access
    async fn test_top_k_results() {
        let wiki = WikipediaQuery::default().with_top_k_results(2);
        let result = wiki.run(json!("programming")).await.unwrap();
        // Count number of "Page:" occurrences
        let page_count = result.matches("Page:").count();
        assert!(page_count <= 2);
    }

    #[tokio::test]
    #[ignore] // Remove ignore to run with network access
    async fn test_nonexistent_page() {
        let wiki = WikipediaQuery::default();
        let result = wiki.run(json!("xyzabc123nonexistentpage999")).await;
        // Should either return no results or handle gracefully
        assert!(result.is_ok());
    }
}
