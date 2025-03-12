use std::{collections::HashMap, error::Error, sync::Arc};

use async_trait::async_trait;
use indoc::indoc;
use reqwest::Client;
use scraper::{Html, Selector};
use serde_json::{json, Value};
use url::Url;

use crate::tools::{
    search::search_result::{SearchResult, SearchResults},
    Tool, ToolFunction, ToolWrapper,
};

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

        let mut query_params = HashMap::new();
        query_params.insert("q", query);

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
            .map(|result| {
                let title = result
                    .select(&result_title_selector)
                    .next()
                    .unwrap()
                    .text()
                    .collect::<Vec<_>>()
                    .join("");
                let link = result
                    .select(&result_url_selector)
                    .next()
                    .unwrap()
                    .text()
                    .collect::<Vec<_>>()
                    .join("")
                    .trim()
                    .to_string();
                let snippet = result
                    .select(&result_snippet_selector)
                    .next()
                    .unwrap()
                    .text()
                    .collect::<Vec<_>>()
                    .join("");

                SearchResult::new(title, link, snippet)
            })
            .take(self.max_results)
            .collect::<Vec<_>>();

        Ok(SearchResults::new(results))
    }
}

const DESCRIPTION: &str = r#"Search the web using Duckduckgo.
Useful for when you need to answer questions about current events.
"#;

#[async_trait]
impl ToolFunction for DuckDuckGoSearch {
    type Input = DuckDuckGoSearchInput;
    type Result = SearchResults;

    fn name(&self) -> String {
        String::from("DuckDuckGoSearch")
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
        let query = input["query"].as_str().ok_or("Invalid input")?;
        Ok(DuckDuckGoSearchInput {
            query: query.to_string(),
        })
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

    #[tokio::test]
    #[ignore]
    async fn duckduckgosearch_tool() {
        let ddg = DuckDuckGoSearch::default().with_max_results(5);
        let s = ddg
            .search("Who is the current President of Peru?")
            .await
            .unwrap();

        println!("{}", s);
    }
}
