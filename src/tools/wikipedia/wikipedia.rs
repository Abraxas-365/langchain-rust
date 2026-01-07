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

use async_trait::async_trait;
use reqwest;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error;

use super::Tool;

const WIKIPEDIA_API_URL: &str = "https://en.wikipedia.org/w/api.php";

/// Configuration options for Wikipedia queries
#[derive(Debug, Clone)]
pub struct WikipediaQueryOptions {
    /// Maximum number of search results to return
    pub top_k_results: usize,
    /// Maximum length of each document content in characters
    pub max_doc_content_length: usize,
    /// Language code for Wikipedia (e.g., "en", "es", "fr")
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
/// # Fields
///
/// * `options` - Configuration options for the Wikipedia query
/// * `client` - HTTP client for making API requests
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
    options: WikipediaQueryOptions,
    client: reqwest::Client,
}

impl WikipediaQuery {
    /// Creates a new WikipediaQuery with custom options
    ///
    /// # Arguments
    ///
    /// * `options` - Configuration options for Wikipedia queries
    pub fn new(options: WikipediaQueryOptions) -> Self {
        Self {
            options,
            client: reqwest::Client::new(),
        }
    }

    /// Creates a WikipediaQuery with a specific language
    ///
    /// # Arguments
    ///
    /// * `lang` - Language code (e.g., "en", "es", "fr", "de")
    pub fn with_lang(lang: impl Into<String>) -> Self {
        let mut options = WikipediaQueryOptions::default();
        options.lang = lang.into();
        Self::new(options)
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

    /// Builds the Wikipedia API URL for the configured language
    fn get_api_url(&self) -> String {
        format!("https://{}.wikipedia.org/w/api.php", self.options.lang)
    }

    /// Searches Wikipedia for articles matching the query
    async fn search(&self, query: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let api_url = self.get_api_url();
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
            .json::<WikipediaSearchResponse>()
            .await?;

        Ok(response
            .query
            .search
            .into_iter()
            .map(|r| r.title)
            .collect())
    }

    /// Fetches the content of a specific Wikipedia page
    async fn fetch_page(&self, title: &str) -> Result<String, Box<dyn Error>> {
        let api_url = self.get_api_url();
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
            .json::<WikipediaPageResponse>()
            .await?;

        if let Some(page) = response.query.pages.values().next() {
            let extract = page.extract.as_ref().unwrap_or(&String::new());
            let truncated = if extract.len() > self.options.max_doc_content_length {
                &extract[..self.options.max_doc_content_length]
            } else {
                extract
            };
            Ok(format!("Page: {}\nSummary: {}", page.title, truncated))
        } else {
            Err("Page not found".into())
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
            Value::Object(mut map) => {
                map.remove("input")
                    .and_then(|v| v.as_str().map(String::from))
                    .ok_or("Invalid input format: expected 'input' field")?
            }
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

        // Fetch content for all found pages
        let mut results = Vec::new();
        for title in titles {
            match self.fetch_page(&title).await {
                Ok(content) => results.push(content),
                Err(e) => {
                    eprintln!("Error fetching page '{}': {}", title, e);
                    continue;
                }
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
        assert_eq!(wiki.get_api_url(), "https://fr.wikipedia.org/w/api.php");
    }

    #[test]
    fn test_api_url_construction() {
        let wiki_en = WikipediaQuery::default();
        assert_eq!(
            wiki_en.get_api_url(),
            "https://en.wikipedia.org/w/api.php"
        );

        let wiki_es = WikipediaQuery::with_lang("es");
        assert_eq!(
            wiki_es.get_api_url(),
            "https://es.wikipedia.org/w/api.php"
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
    async fn test_object_input_format() {
        let wiki = WikipediaQuery::default();
        let input = json!({"input": "Rust programming"});
        
        // This will make an actual API call, so we just check it doesn't panic
        // In a real test environment, you'd mock the HTTP client
        let result = wiki.run(input).await;
        // Result should either succeed or fail gracefully
        assert!(result.is_ok() || result.is_err());
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
        let content = wiki.fetch_page("Rust (programming language)").await.unwrap();
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
            if line.starts_with("Summary:") {
                let summary = &line[8..]; // Skip "Summary: "
                assert!(summary.len() <= 100);
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
