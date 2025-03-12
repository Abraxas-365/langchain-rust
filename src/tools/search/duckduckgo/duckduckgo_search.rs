use std::{collections::HashMap, error::Error, sync::Arc};

use async_trait::async_trait;
use derive_new::new;
use indoc::indoc;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use url::Url;

use crate::tools::{
    search::search_result::{SearchResult, SearchResults},
    Tool, ToolFunction, ToolWrapper,
};

#[derive(Deserialize, Serialize, new)]
pub struct DuckDuckGoSearchInput {
    query: String,
}

pub struct DuckDuckGoSearch {
    url: String,
    client: Client,
    max_results: usize,
}

impl DuckDuckGoSearch {
    pub fn with_max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }

    pub async fn search(&self, query: &str) -> Result<SearchResults, Box<dyn Error>> {
        let mut url = Url::parse(&self.url)?;

        let query_params = HashMap::from([("q", query)]);
        url.query_pairs_mut().extend_pairs(query_params.iter());

        let response = self.client.get(url).send().await?;
        let body = response.text().await?;
        let document = Html::parse_document(&body);

        let result_selector = Selector::parse(".web-result").unwrap();
        let result_title_selector = Selector::parse(".result__a").unwrap();
        let result_url_selector = Selector::parse(".result__url").unwrap();
        let result_snippet_selector = Selector::parse(".result__snippet").unwrap();

        let results = document
            .select(&result_selector)
            .filter_map(|result| {
                let title = result
                    .select(&result_title_selector)
                    .next()?
                    .text()
                    .collect::<Vec<_>>()
                    .join("");
                let link = result
                    .select(&result_url_selector)
                    .next()?
                    .text()
                    .collect::<Vec<_>>()
                    .join("")
                    .trim()
                    .to_string();
                let snippet = result
                    .select(&result_snippet_selector)
                    .next()?
                    .text()
                    .collect::<Vec<_>>()
                    .join("");

                Some(SearchResult::new(title, link, snippet))
            })
            .take(self.max_results)
            .collect::<Vec<_>>();

        Ok(SearchResults::new(results))
    }
}

const DESCRIPTION: &str = r#"Search the web using Duckduckgo.
Useful for when you need to answer questions about current events."#;

#[async_trait]
impl ToolFunction for DuckDuckGoSearch {
    type Input = DuckDuckGoSearchInput;
    type Result = SearchResults;

    fn name(&self) -> String {
        "DuckDuckGo Search".to_string()
    }

    fn description(&self) -> String {
        format!(
            "{}\n{}",
            DESCRIPTION,
            indoc! {"
                The input for this tool MUST be in the following format:
                {{
                    query (String): The query you want to search for,
                }}
            "}
        )
    }

    fn parameters(&self) -> Value {
        json!({
            "description": DESCRIPTION,
            "type": "string",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to look up"
                }
            },
            "required": ["query"]
        })
    }

    async fn parse_input(&self, input: Value) -> Result<Self::Input, Box<dyn Error>> {
        let result = serde_json::from_value::<DuckDuckGoSearchInput>(input.clone())
            .or_else(|_| serde_json::from_value::<String>(input).map(DuckDuckGoSearchInput::new))?;

        Ok(result)
    }

    async fn run(&self, input: DuckDuckGoSearchInput) -> Result<SearchResults, Box<dyn Error>> {
        self.search(&input.query).await
    }
}

impl Default for DuckDuckGoSearch {
    fn default() -> Self {
        Self {
            client: Client::new(),
            url: "https://duckduckgo.com/html/".to_string(),
            max_results: 4,
        }
    }
}

impl From<DuckDuckGoSearch> for Arc<dyn Tool> {
    fn from(val: DuckDuckGoSearch) -> Self {
        Arc::new(ToolWrapper::new(val))
    }
}

#[cfg(test)]
mod tests {
    use super::DuckDuckGoSearch;
    use crate::tools::Tool;
    use serde_json::json;
    use std::sync::Arc;

    #[tokio::test]
    #[ignore]
    async fn duckduckgosearch_tool() {
        let tool: Arc<dyn Tool> = DuckDuckGoSearch::default().with_max_results(5).into();
        let input = json!({
            "query": "Who is the current President of Peru?"
        });

        let result = tool.call(input).await.unwrap();

        println!("{}", result);
    }

    #[tokio::test]
    #[ignore]
    async fn duckduckgosearch_tool_empty() {
        let tool: Arc<dyn Tool> = DuckDuckGoSearch::default().into();
        let input = json!({
            "query": "vaygbuoipqyngxaupoidfcaasdcfjlkqwhfqhsakdnasfsfclkvahsxczkgjqeopjraoisphd"
        });

        let result = tool.call(input).await.unwrap();

        println!("{}", result);
    }
}
