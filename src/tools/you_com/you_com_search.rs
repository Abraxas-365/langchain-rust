use std::error::Error;

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::tools::Tool;

/// Wrapper around the You.com Web Search API.
///
/// Requires an API key from <https://you.com/platform/api-keys>, either passed
/// to [`YouComSearch::new`] or read from the `YDC_API_KEY` environment variable
/// when using [`YouComSearch::default`].
pub struct YouComSearch {
    api_key: String,
    client: Client,
    count: usize,
}

impl YouComSearch {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::new(),
            count: 5,
        }
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// Sets the maximum number of results to return (default 5).
    pub fn with_count(mut self, count: usize) -> Self {
        self.count = count;
        self
    }

    pub async fn search(&self, query: &str) -> Result<String, Box<dyn Error>> {
        let response = self
            .client
            .post("https://ydc-index.io/v1/search")
            .header("X-API-Key", &self.api_key)
            .json(&json!({
                "query": query,
                "count": self.count,
            }))
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(format!("You.com search failed: {} - {}", status, body).into());
        }

        let parsed: Value = serde_json::from_str(&body)?;
        let results = extract_results(&parsed)?;

        Ok(serde_json::to_string(&results)?)
    }
}

fn extract_results(response: &Value) -> Result<Vec<YouComSearchResult>, Box<dyn Error>> {
    let hits = response["results"]["web"]
        .as_array()
        .ok_or("Unexpected You.com response: missing results.web")?;

    let results = hits
        .iter()
        .map(|hit| YouComSearchResult {
            title: hit["title"].as_str().unwrap_or_default().to_string(),
            url: hit["url"].as_str().unwrap_or_default().to_string(),
            description: hit["description"].as_str().unwrap_or_default().to_string(),
        })
        .collect();

    Ok(results)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YouComSearchResult {
    title: String,
    url: String,
    description: String,
}

#[async_trait]
impl Tool for YouComSearch {
    fn name(&self) -> String {
        String::from("YouComSearch")
    }

    fn description(&self) -> String {
        String::from(
            "A wrapper around the You.com Web Search API. \
            Useful for when you need to answer questions about current events. \
            Always one of the first options when you need to find information on internet. \
            Input should be a search query. Output is a JSON array of the query results.",
        )
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let input = input.as_str().ok_or("Input should be a string")?;
        self.search(input).await
    }

    fn parameters(&self) -> Value {
        let prompt = "A wrapper around the You.com Web Search API. \
            Useful for when you need to answer questions about current events. \
            Input should be a search query. Output is a JSON array of the query results.";

        json!({
            "description": prompt,
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to look up"
                }
            },
            "required": ["query"]
        })
    }
}

impl Default for YouComSearch {
    fn default() -> YouComSearch {
        YouComSearch {
            api_key: std::env::var("YDC_API_KEY").unwrap_or_default(),
            client: Client::new(),
            count: 5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{extract_results, YouComSearch};
    use serde_json::json;

    #[test]
    fn parses_youcom_response() {
        let response = json!({
            "results": {
                "web": [
                    {
                        "title": "Rust Programming Language",
                        "url": "https://www.rust-lang.org/",
                        "description": "A language empowering everyone to build reliable software."
                    },
                    {
                        "title": "Example Domain",
                        "url": "https://example.com/",
                        "description": "Missing optional fields default to empty strings."
                    }
                ]
            }
        });

        let results = extract_results(&response).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].title, "Rust Programming Language");
        assert_eq!(results[0].url, "https://www.rust-lang.org/");
    }

    #[test]
    fn rejects_unexpected_response_shape() {
        let response = json!({"error": "something else"});
        assert!(extract_results(&response).is_err());
    }

    #[tokio::test]
    #[ignore]
    async fn youcom_search_tool() {
        let youcom = YouComSearch::default().with_count(5);
        let s = youcom
            .search("Who is the current President of Peru?")
            .await
            .unwrap();

        println!("{}", s);
    }
}
