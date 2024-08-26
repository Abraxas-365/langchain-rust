use std::{io::Read, path::Path, pin::Pin};

use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use pdf_extract::{output_doc, PlainTextOutput};

use crate::{
    document_loaders::{process_doc_stream, Loader, LoaderError},
    schemas::Document,
    text_splitter::TextSplitter,
};

#[derive(Debug, Clone)]
pub struct PdfExtractLoader {
    document: pdf_extract::Document,
}

impl PdfExtractLoader {
    /// Creates a new PdfLoader from anything that implements the Read trait.
    /// This is a generic constructor which can be used with any type of reader.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use std::io::Cursor;
    /// let data = Cursor::new(vec![...] /* some PDF data */);
    /// let loader = PdfExtractLoader::new(data)?;
    /// ```
    ///
    pub fn new<R: Read>(reader: R) -> Result<Self, LoaderError> {
        let document = pdf_extract::Document::load_from(reader)?;
        Ok(Self { document })
    }
    /// Creates a new PdfLoader from a path to a PDF file.
    /// This loads the PDF document and creates a PdfLoader from it.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let loader = PdfExtractLoader::from_path("/path/to/my.pdf")?;
    /// ```
    ///
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, LoaderError> {
        let document = pdf_extract::Document::load(path)?;
        Ok(Self { document })
    }
}

#[async_trait]
impl Loader for PdfExtractLoader {
    async fn load(
        mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let mut buffer: Vec<u8> = Vec::new();
        let mut output = PlainTextOutput::new(&mut buffer as &mut dyn std::io::Write);
        output_doc(&self.document, &mut output)?;
        let doc = Document::new(String::from_utf8(buffer)?);

        let stream = stream! {
            yield Ok(doc);
        };

        Ok(Box::pin(stream))
    }

    async fn load_and_split<TS: TextSplitter + 'static>(
        mut self,
        splitter: TS,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let doc_stream = self.load().await?;
        let stream = process_doc_stream(doc_stream, splitter).await;
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::Cursor};

    use futures_util::StreamExt;

    use super::*;

    #[tokio::test]
    async fn test_lo_pdf_loader() {
        let path = "./src/document_loaders/test_data/sample.pdf";

        let loader = PdfExtractLoader::from_path(path).expect("Failed to create PdfExtractLoader");

        let docs = loader
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;

        assert_eq!(&docs[0].page_content[..100], "\n\nSample PDF Document\n\nRobert Maron\nGrzegorz Grudzi´nski\n\nFebruary 20, 1999\n\n2\n\nContents\n\n1 Templat");
        assert_eq!(docs.len(), 1);
    }

    #[tokio::test]
    async fn test_lo_pdf_loader_reader() {
        let path = "./src/document_loaders/test_data/sample.pdf";
        let mut file = File::open(path).unwrap();
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).unwrap();
        let reader = Cursor::new(buffer);

        let loader = PdfExtractLoader::new(reader).expect("Failed to create PdfExtractLoader");

        let docs = loader
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;

        assert_eq!(&docs[0].page_content[..100], "\n\nSample PDF Document\n\nRobert Maron\nGrzegorz Grudzi´nski\n\nFebruary 20, 1999\n\n2\n\nContents\n\n1 Templat");
        assert_eq!(docs.len(), 1);
    }
}
