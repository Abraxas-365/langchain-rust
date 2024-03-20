use async_trait::async_trait;

use crate::{schemas::Document, text_splitter::TextSplitter};

use super::LoaderError;

#[async_trait]
pub trait Loader: Send + Sync {
    async fn load(mut self) -> Result<Vec<Document>, LoaderError>;
    async fn load_and_split<TS: TextSplitter + 'static>(
        mut self,
        splitter: TS,
    ) -> Result<Vec<Document>, LoaderError>;
}
