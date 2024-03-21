use std::error::Error;

use async_trait::async_trait;
use serde_json::Value;

use crate::tools::Tool;

pub struct SerpApi {
    api_key: String,
    location: Option<String>,
    hl: Option<String>,
    gl: Option<String>,
    google_domain: Option<String>,
}

impl SerpApi {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            location: None,
            hl: None,
            gl: None,
            google_domain: None,
        }
    }
    pub fn with_location<S: Into<String>>(mut self, location: S) -> Self {
        self.location = Some(location.into());
        self
    }
    pub fn with_hl<S: Into<String>>(mut self, hl: S) -> Self {
        self.hl = Some(hl.into());
        self
    }
    pub fn with_gl(mut self, gl: String) -> Self {
        self.gl = Some(gl);
        self
    }
    pub fn with_google_domain<S: Into<String>>(mut self, google_domain: S) -> Self {
        self.google_domain = Some(google_domain.into());
        self
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = api_key.into();
        self
    }

    pub async fn simple_search(&self, query: &str) -> Result<String, Box<dyn Error>> {
        let mut url = format!(
            "https://serpapi.com/search.json?q={}&api_key={}",
            query, self.api_key
        );
        if let Some(location) = &self.location {
            url.push_str(&format!("&location={}", location));
        }
        if let Some(hl) = &self.hl {
            url.push_str(&format!("&hl={}", hl));
        }
        if let Some(gl) = &self.gl {
            url.push_str(&format!("&gl={}", gl));
        }
        if let Some(google_domain) = &self.google_domain {
            url.push_str(&format!("&google_domain={}", google_domain));
        }
        let results: Value = reqwest::get(&url).await?.json().await?;

        let res = process_response(&results)?;

        Ok(res)
    }
}

fn get_answer_box(result: &Value) -> String {
    if let Some(map) = result["answer_box"].as_object() {
        if let Some(answer) = map.get("answer").and_then(|v| v.as_str()) {
            return answer.to_string();
        }

        if let Some(snippet) = map.get("snippet").and_then(|v| v.as_str()) {
            return snippet.to_string();
        }

        if let Some(snippet) = map
            .get("snippet_highlighted_words")
            .and_then(|v| v.as_array())
        {
            if !snippet.is_empty() {
                if let Some(first) = snippet.first().and_then(|v| v.as_str()) {
                    return first.to_string();
                }
            }
        }
    }

    "".to_string()
}

fn process_response(res: &Value) -> Result<String, Box<dyn Error>> {
    if !get_answer_box(res).is_empty() {
        return Ok(get_answer_box(res));
    }
    if !get_sport_result(res).is_empty() {
        return Ok(get_sport_result(res));
    }
    if !get_knowledge_graph(res).is_empty() {
        return Ok(get_knowledge_graph(res));
    }
    if !get_organic_result(res).is_empty() {
        return Ok(get_organic_result(res));
    }
    Err("No good result".into())
}

fn get_sport_result(result: &Value) -> String {
    if let Some(map) = result["sports_results"].as_object() {
        if let Some(game_spotlight) = map.get("game_spotlight").and_then(|v| v.as_str()) {
            return game_spotlight.to_string();
        }
    }

    "".to_string()
}

fn get_knowledge_graph(result: &Value) -> String {
    if let Some(map) = result["knowledge_graph"].as_object() {
        if let Some(description) = map.get("description").and_then(|v| v.as_str()) {
            return description.to_string();
        }
    }

    "".to_string()
}

fn get_organic_result(result: &Value) -> String {
    if let Some(array) = result["organic_results"].as_array() {
        if !array.is_empty() {
            if let Some(first) = array.first() {
                if let Some(first_map) = first.as_object() {
                    if let Some(snippet) = first_map.get("snippet").and_then(|v| v.as_str()) {
                        return snippet.to_string();
                    }
                }
            }
        }
    }

    "".to_string()
}

#[async_trait]
impl Tool for SerpApi {
    fn name(&self) -> String {
        String::from("GoogleSearch")
    }
    fn description(&self) -> String {
        String::from(
            r#""A wrapper around Google Search. "
	"Useful for when you need to answer questions about current events. "
	"Always one of the first options when you need to find information on internet"
	"Input should be a search query."#,
        )
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let input = input.as_str().ok_or("Input should be a string")?;
        self.simple_search(input).await
    }
}

impl Default for SerpApi {
    fn default() -> SerpApi {
        SerpApi {
            api_key: std::env::var("SERPAPI_API_KEY").unwrap_or_default(),
            location: None,
            hl: None,
            gl: None,
            google_domain: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SerpApi;

    #[tokio::test]
    #[ignore]
    async fn serpapi_tool() {
        let serpapi = SerpApi::default();
        let s = serpapi
            .simple_search("Who is the President of Peru")
            .await
            .unwrap();
        println!("{}", s);
    }
}
