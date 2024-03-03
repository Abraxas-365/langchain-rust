use async_trait::async_trait;
use regex::Regex;
use scraper::{ElementRef, Html, Selector};
use std::{error::Error, sync::Arc};

use crate::tools::Tool;

pub struct WebScrapper {}

impl WebScrapper {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait]
impl Tool for WebScrapper {
    fn name(&self) -> String {
        String::from("Web Scraper")
    }
    fn description(&self) -> String {
        String::from(
            "Web Scraper will scan a url and return the content of the web page.
		Input should be a working url.",
        )
    }
    async fn call(&self, input: &str) -> Result<String, Box<dyn Error>> {
        match scrape_url(input).await {
            Ok(content) => Ok(content),
            Err(e) => Ok(format!("Error scraping {}: {}\n", input, e)),
        }
    }
}

impl Into<Arc<dyn Tool>> for WebScrapper {
    fn into(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }
}

async fn scrape_url(url: &str) -> Result<String, Box<dyn Error>> {
    let res = reqwest::get(url).await?.text().await?;

    let document = Html::parse_document(&res);
    let body_selector = Selector::parse("body").unwrap();

    let mut text = Vec::new();
    for element in document.select(&body_selector) {
        collect_text_not_in_script(&element, &mut text);
    }

    let joined_text = text.join(" ");
    let cleaned_text = joined_text.replace("\n", " ").replace("\t", " ");
    let re = Regex::new(r"\s+").unwrap();
    let final_text = re.replace_all(&cleaned_text, " ");
    Ok(final_text.to_string())
}

fn collect_text_not_in_script(element: &ElementRef, text: &mut Vec<String>) {
    for node in element.children() {
        if node.value().is_element() {
            let tag_name = node.value().as_element().unwrap().name();
            if tag_name == "script" {
                continue;
            }
            collect_text_not_in_script(&ElementRef::wrap(node).unwrap(), text);
        } else if node.value().is_text() {
            text.push(node.value().as_text().unwrap().text.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;

    #[tokio::test]
    async fn test_scrape_url() {
        // Request a new server from the pool
        let mut server = mockito::Server::new_async().await;

        // Create a mock on the server
        let mock = server
            .mock("GET", "/")
            .with_status(200)
            .with_header("content-type", "text/plain")
            .with_body("<html><body>Hello World</body></html>")
            .create();

        // Instantiate your WebScrapper
        let scraper = WebScrapper::new();

        // Use the server URL for scraping
        let url = server.url();

        // Call the WebScrapper with the mocked URL
        let result = scraper.call(&url).await;

        // Assert that the result is Ok and contains "Hello World"
        assert!(result.is_ok());
        let content = result.unwrap();
        assert_eq!(content.trim(), "Hello World");

        // Verify that the mock was called as expected
        mock.assert();
    }
}
