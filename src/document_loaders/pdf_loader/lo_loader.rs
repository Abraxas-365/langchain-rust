use std::{collections::HashMap, io::Read, path::Path};

use async_trait::async_trait;
use serde_json::Value;

use crate::{
    document_loaders::{Loader, LoaderError},
    schemas::Document,
    text_splitter::TextSplitter,
};

#[derive(Debug, Clone)]
pub struct LoPdfLoader {
    document: lopdf::Document,
}

impl LoPdfLoader {
    /// Creates a new PdfLoader from anything that implements the Read trait.
    /// This is a generic constructor which can be used with any type of reader.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::io::Cursor;
    /// let data = Cursor::new(vec![...] /* some PDF data */);
    /// let loader = PdfLoader::new(data)?;
    /// ```
    ///
    pub fn new<R: Read>(reader: R) -> Result<Self, LoaderError> {
        let document = lopdf::Document::load_from(reader)?;
        Ok(Self { document })
    }
    /// Creates a new PdfLoader from a path to a PDF file.
    /// This loads the PDF document and creates a PdfLoader from it.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loader = PdfLoader::from_path("/path/to/my.pdf")?;
    /// ```
    ///
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, LoaderError> {
        let document = lopdf::Document::load(path)?;
        Ok(Self { document })
    }
}

#[async_trait]
impl Loader for LoPdfLoader {
    async fn load(mut self) -> Result<Vec<Document>, LoaderError> {
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
    ) -> Result<Vec<Document>, LoaderError> {
        let documents = self.load().await?;
        let result = splitter.split_documents(&documents)?;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Cursor};

    use super::*;

    #[tokio::test]
    async fn test_pdf_loader() {
        let path = "./src/document_loaders/test_data/sample.pdf";

        let loader = LoPdfLoader::from_path(path).expect("Failed to create PdfLoader");

        let docs = loader.load().await.expect("Failed to load content");

        assert_eq!(
            docs[0].page_content,
            "Sample PDF Document Robert Maron Grzegorz Grudzi · nski February 20, 1999 \n"
        );
        assert_eq!(docs.len(), 10);
    }

    #[tokio::test]
    async fn test_lo_pdf_loader_reader() {
        let path = "./src/document_loaders/test_data/sample.pdf";
        let mut file = File::open(path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        let reader = Cursor::new(buffer);

        let loader = LoPdfLoader::new(reader).expect("Failed to create PdfLoader");

        let docs = loader.load().await.expect("Failed to load content");

        assert_eq!(
            docs[0].page_content,
            "Sample PDF Document Robert Maron Grzegorz Grudzi · nski February 20, 1999 \n"
        );
        assert_eq!(docs.len(), 10);
    }
}
