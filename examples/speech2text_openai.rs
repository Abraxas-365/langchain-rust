use std::error::Error;

use async_trait::async_trait;
use langchain_rust::tools::{SpeechStorage, Text2SpeechOpenAI, Tool};
use serde_json::Value;

#[allow(dead_code)]
struct XStorage {}

//You can add save te result to s3 or other storage using

#[async_trait]
impl SpeechStorage for XStorage {
    async fn save(&self, path: &str, _data: &[u8]) -> Result<String, Box<dyn Error + Send + Sync>> {
        println!("Saving to: {}", path);
        Ok(path.to_string())
    }
}

#[tokio::main]
async fn main() {
    let openai = Text2SpeechOpenAI::default().with_path("./data/audio.mp3");
    // .with_storage(XStorage {});

    let path = openai
        .call(Value::String("Hi, My name is Luis".to_string()))
        .await
        .unwrap();
    println!("Path: {}", path);
}
