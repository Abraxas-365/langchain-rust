use std::{collections::HashMap, io::Read, path::Path, pin::Pin};

use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use serde_json::Value;

use crate::{
    document_loaders::{process_doc_stream, Loader, LoaderError},
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
    /// let loader = LoPdfLoader::new(data)?;
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
    /// let loader = LoPdfLoader::from_path("/path/to/my.pdf")?;
    /// ```
    ///
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self, LoaderError> {
        let document = lopdf::Document::load(path)?;
        Ok(Self { document })
    }
}

#[async_trait]
impl Loader for LoPdfLoader {
    async fn load(
        mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let document = std::sync::Arc::new(self.document);
        let stream = stream! {
            let pages = document.get_pages();
            for (i, _) in pages.iter().enumerate() {
                let page_number = (i + 1) as u32;
                // Text extraction is CPU-bound: one page at a time on the blocking pool.
                let page = document.clone();
                let text = crate::blocking::run_blocking(move || page.extract_text(&[page_number])).await?;
                let mut metadata = HashMap::new();
                metadata.insert("page_number".to_string(), Value::from(page_number));
                let doc=Document::new(text).with_metadata(metadata);
                yield Ok(doc);

            }
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

        let loader = LoPdfLoader::from_path(path).expect("Failed to create LoPdfLoader");

        let docs = loader
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;

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

        let loader = LoPdfLoader::new(reader).expect("Failed to create LoPdfLoader");

        let docs = loader
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;

        assert_eq!(
            docs[0].page_content,
            "Sample PDF Document Robert Maron Grzegorz Grudzi · nski February 20, 1999 \n"
        );
        assert_eq!(docs.len(), 10);
    }

    /// Pages are extracted on the blocking pool, in order, without stalling the runtime
    /// thread (#142).
    #[tokio::test]
    async fn lo_pdf_loader_does_not_block_the_runtime() {
        let loader = LoPdfLoader::from_path("./src/document_loaders/test_data/sample.pdf")
            .expect("Failed to create LoPdfLoader");
        let (docs, ticks) = crate::blocking::testing::ticks_while(async {
            loader
                .load()
                .await
                .unwrap()
                .map(|d| d.unwrap())
                .collect::<Vec<_>>()
                .await
        })
        .await;
        let pages: Vec<_> = docs
            .iter()
            .map(|d| d.metadata["page_number"].clone())
            .collect();
        assert_eq!(pages, (1..=10).map(Value::from).collect::<Vec<_>>());
        assert!(ticks > 0, "PDF extraction blocked the runtime thread");
    }
}
