use std::error::Error;

use async_trait::async_trait;
use tokio::io::{AsyncRead, AsyncReadExt, BufReader};

use crate::{document_loaders::Loader, schemas::Document};

pub struct TextLoader<R: AsyncRead> {
    pub(crate) reader: R,
}

impl<R: AsyncRead> TextLoader<R> {
    pub fn new(reader: R) -> Self {
        Self { reader }
    }
}

#[async_trait]
impl<R: AsyncRead + Unpin + Send + Sync> Loader for TextLoader<R> {
    async fn load(&mut self) -> Result<Vec<Document>, Box<dyn Error>> {
        let mut reader = BufReader::new(&mut self.reader);
        let mut buffer = Vec::new();

        reader.read_to_end(&mut buffer).await?;

        let buffer_string = std::str::from_utf8(&buffer)?.to_owned();

        Ok(vec![Document::new(buffer_string)])
    }

    async fn load_and_split<TS: crate::text_splitter::TextSplitter + 'static>(
        &mut self,
        splitter: TS,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let documents = self.load().await?;
        let result = splitter.split_documents(&documents)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    #[tokio::test]
    async fn test_load() {
        let data = b"Hello, world!";
        let reader = Cursor::new(data as &[u8]);
        let mut loader = crate::document_loaders::TextLoader { reader };

        let result = crate::document_loaders::Loader::load(&mut loader).await;

        match result {
            Ok(docs) => {
                assert_eq!(docs.len(), 1);
                assert_eq!(docs[0].page_content, "Hello, world!");
            }
            Err(e) => {
                panic!("Error during loading: {}", e);
            }
        }
    }
}
