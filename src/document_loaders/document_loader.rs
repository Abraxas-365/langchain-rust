use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::{schemas::Document, text_splitter::TextSplitter};

use super::LoaderError;

#[async_trait]
pub trait Loader: Send + Sync {
    async fn load(
        self,
    ) -> Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + Sync + 'static>>;
    async fn load_and_split<TS: TextSplitter + 'static>(
        self,
        splitter: TS,
    ) -> Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + Sync + 'static>>;
}
