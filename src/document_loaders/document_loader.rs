use std::error::Error;

use async_trait::async_trait;

use crate::{schemas::Document, text_splitter::TextSplitter};

#[async_trait]
pub trait Loader: Send + Sync {
    async fn load(mut self) -> Result<Vec<Document>, Box<dyn Error>>;
    async fn load_and_split<TS: TextSplitter + 'static>(
        mut self,
        splitter: TS,
    ) -> Result<Vec<Document>, Box<dyn Error>>;
}
