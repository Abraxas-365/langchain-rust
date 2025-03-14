use std::error::Error;

use async_trait::async_trait;

#[async_trait]
pub trait SpeechStorage: Send + Sync {
    async fn save(&self, key: &str, data: &[u8]) -> Result<String, Box<dyn Error + Send + Sync>>;
}
