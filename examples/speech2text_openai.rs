use async_trait::async_trait;
use langchain_rust::tools::{SpeechStorage, Text2SpeechOpenAI, Tool};

#[allow(dead_code)]
struct XStorage {}

//You can add save te result to s3 or other storage using

#[async_trait]
impl SpeechStorage for XStorage {
    async fn save(&self, path: &str, _data: &[u8]) -> Result<String, Box<dyn std::error::Error>> {
        println!("Saving to: {}", path);
        Ok(path.to_string())
    }
}

#[tokio::main]
async fn main() {
    let openai = Text2SpeechOpenAI::default().with_path("./data/audio.mp3");
    // .with_storage(XStorage {});

    let path = openai.call("Hi, My name is Luis").await.unwrap();
    println!("Path: {}", path);
}
