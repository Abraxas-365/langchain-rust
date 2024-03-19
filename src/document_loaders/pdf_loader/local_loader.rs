use std::{
    collections::HashMap,
    error::Error,
    fs::File,
    io::{Cursor, Read},
    path::Path,
};

use async_trait::async_trait;
use serde_json::Value;

use crate::{document_loaders::Loader, schemas::Document, text_splitter::TextSplitter};

#[derive(Debug, Clone)]
pub struct PdfLoader {
    document: lopdf::Document,
}

impl PdfLoader {
    pub fn new<R: Read>(reader: R) -> Result<Self, Box<dyn Error>> {
        let document = lopdf::Document::load_from(reader)?;
        Ok(Self { document })
    }

    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let reader = Cursor::new(buffer);
        let document = lopdf::Document::load_from(reader.clone())?;
        Ok(Self { document })
    }

    pub fn from_string<S: Into<String>>(input: S) -> Result<Self, Box<dyn Error>> {
        let input = input.into();
        let reader = Cursor::new(input.into_bytes());
        let document = lopdf::Document::load_from(reader.clone())?;
        Ok(Self { document })
    }
}

#[async_trait]
impl Loader for PdfLoader {
    async fn load(mut self) -> Result<Vec<Document>, Box<dyn Error>> {
        let mut documents: Vec<Document> = Vec::new();
        let pages = self.document.get_pages();
        for (i, _) in pages.iter().enumerate() {
            let page_number = (i + 1) as u32;
            let text = self.document.extract_text(&[page_number])?;
            let mut metadata = HashMap::new();
            metadata.insert("page_number".to_string(), Value::from(page_number));
            documents.push(Document::new(text).with_metadata(metadata))
        }

        Ok(documents)
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
    async fn test_pdf_loader() {
        let path = "./src/document_loaders/test_data/sample.pdf";

        let loader = PdfLoader::from_path(path).expect("Failed to create PdfLoader");

        let docs = loader.load().await.expect("Failed to load content");

        assert_eq!(
            docs[0].page_content,
            "Sample PDF Document Robert Maron Grzegorz Grudzi · nski February 20, 1999 \n"
        );
        assert_eq!(docs.len(), 10);
    }
}
