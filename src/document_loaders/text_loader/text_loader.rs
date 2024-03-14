use std::error::Error;

use async_trait::async_trait;

use crate::{document_loaders::Loader, schemas::Document, text_splitter::TextSplitter};

#[derive(Debug, Clone)]
pub struct TextLoader {
    content: String,
}

impl TextLoader {
    pub fn new<T: Into<String>>(input: T) -> Self {
        Self {
            content: input.into(),
        }
    }
}

#[async_trait]
impl Loader for TextLoader {
    async fn load(mut self) -> Result<Vec<Document>, Box<dyn Error>> {
        Ok(vec![Document::new(self.content.clone())])
    }

    async fn load_and_split<TS: TextSplitter + 'static>(
        mut self,
        splitter: TS,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let documents = self.load().await?;
        let result = splitter.split_documents(&documents)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reading_mocked_file_content() {
        let mocked_file_content = "This is the content of the mocked file.";

        // Create a new TextLoader with the mocked content
        let loader = TextLoader::new(mocked_file_content.to_string());

        // Use the loader to load the content, which should be wrapped in a Document
        let documents = loader.load().await.expect("Failed to load content");

        assert_eq!(documents.len(), 1); // Assuming the content should be wrapped in a single Document
        assert_eq!(documents[0].page_content, mocked_file_content); // Ensure the Document contains the mocked content
    }
}
