use async_trait::async_trait;

use crate::tools::Tool;
use std::error::Error;

#[derive(serde::Serialize, serde::Deserialize)]
struct WolframResponse {
    queryresult: WolframResponseContent,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct WolframResponseContent {
    pods: Vec<Pod>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Pod {
    title: String,
    subpods: Vec<Subpod>,
}

impl From<Pod> for String {
    fn from(pod: Pod) -> String {
        let subpods_str: Vec<String> = pod
            .subpods
            .into_iter()
            .filter_map(|subpod| {
                let subpod_str = String::from(subpod);
                if subpod_str.is_empty() {
                    return None;
                }
                Some(subpod_str)
            })
            .collect();

        if subpods_str.is_empty() {
            return String::from("");
        }

        format!(
            "{{\"title\": {},\"subpods\": [{}]}}",
            pod.title,
            subpods_str.join(",")
        )
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Subpod {
    title: String,
    plaintext: String,
}

impl From<Subpod> for String {
    fn from(subpod: Subpod) -> String {
        if subpod.plaintext.is_empty() {
            return String::from("");
        }

        format!(
            "{{\"title\": \"{}\",\"plaintext\": \"{}\"}}",
            subpod.title,
            subpod.plaintext.replace("\n", " // ")
        )
    }
}

pub struct Wolfram {
    app_id: String,
    exclude_pods: Vec<String>,
}

impl Wolfram {
    pub fn new(app_id: String) -> Self {
        Self {
            app_id,
            exclude_pods: Vec::new(),
        }
    }

    pub fn with_excludes(mut self, exclude_pods: Vec<String>) -> Self {
        self.exclude_pods = exclude_pods;
        self
    }
}

impl Default for Wolfram {
    fn default() -> Wolfram {
        Wolfram {
            app_id: std::env::var("WOLFRAM_APP_ID").unwrap_or(String::new()),
            exclude_pods: Vec::new(),
        }
    }
}

#[async_trait]
impl Tool for Wolfram {
    fn name(&self) -> String {
        String::from("Wolfram")
    }

    fn description(&self) -> String {
        String::from(
            "Wolfram Solver leverages the Wolfram Alpha computational engine
            to solve complex queries. Input should be a valid mathematical 
            expression or query formulated in a way that Wolfram Alpha can 
            interpret.",
        )
    }
    async fn call(&self, input: &str) -> Result<String, Box<dyn Error>> {
        let mut url = format!(
            "https://api.wolframalpha.com/v2/query?appid={}&input={}&output=JSON&format=plaintext&podstate=Result__Step-by-step+solution",
            self.app_id,
            urlencoding::encode(input)
        );

        if !self.exclude_pods.is_empty() {
            url += &format!("&excludepodid={}", self.exclude_pods.join(","));
        }

        let response_text = reqwest::get(&url).await?.text().await?;
        let response: WolframResponse = serde_json::from_str(&response_text)?;

        let pods_str: Vec<String> = response
            .queryresult
            .pods
            .into_iter()
            .filter_map(|pod| {
                let pod_str = String::from(pod);
                if pod_str.is_empty() {
                    return None;
                }
                Some(pod_str)
            })
            .collect();

        Ok(format!("{{\"pods\": [{}]}}", pods_str.join(",")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test as async_test;

    #[async_test]
    async fn test_wolfram() {
        let wolfram = Wolfram::default().with_excludes(vec!["Plot".to_string()]);
        let input = "solve x^2 = 4";
        let result = wolfram.call(input).await;

        assert!(result.is_ok());
        println!("{}", result.unwrap());
    }
}
