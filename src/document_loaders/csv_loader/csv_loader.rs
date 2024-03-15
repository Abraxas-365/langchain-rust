use crate::{document_loaders::Loader, schemas::Document, text_splitter::TextSplitter};
use async_trait::async_trait;
use csv;
use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::io::{Cursor, Read};
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncReadExt;

#[derive(Debug, Clone)]
pub struct CsvLoader<R> {
    reader: R,
    columns: Vec<String>,
}

impl<R: Read> CsvLoader<R> {
    pub fn new(reader: R, columns: Vec<String>) -> Self {
        Self { reader, columns }
    }
}

impl CsvLoader<Cursor<Vec<u8>>> {
    pub fn from_string<S: Into<String>>(input: S, columns: Vec<String>) -> Self {
        let input = input.into();
        let reader = Cursor::new(input.into_bytes());
        Self::new(reader, columns)
    }
}

impl CsvLoader<Cursor<Vec<u8>>> {
    pub async fn from_path<P: AsRef<Path>>(
        path: P,
        columns: Vec<String>,
    ) -> Result<Self, Box<dyn Error>> {
        let mut file = File::open(path).await?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await?;
        let reader = Cursor::new(buffer);
        Ok(Self::new(reader, columns))
    }
}

#[async_trait]
impl<R: Read + Send + Sync + 'static> Loader for CsvLoader<R> {
    async fn load(mut self) -> Result<Vec<Document>, Box<dyn Error>> {
        let mut reader = csv::Reader::from_reader(self.reader);
        let mut documents = vec![];
        let headers = reader.headers()?.clone();

        // Initialize rown to track row number
        let mut row_number: i64 = 0;

        for result in reader.records() {
            let record = result?;
            let mut content = String::new();

            for (i, field) in record.iter().enumerate() {
                let header = &headers[i];
                if !self.columns.contains(&header.to_string()) {
                    continue;
                }

                let line = format!("{}: {}", header, field);
                content.push_str(&line);
                content.push('\n');
            }

            row_number += 1; // Increment the row number by 1 for each row

            // Generate document with the content and metadata
            let mut document = Document::new(content);
            let mut metadata = HashMap::new();
            metadata.insert("row".to_string(), Value::from(row_number));

            // Attach the metadata to the document
            document.metadata = metadata;

            documents.push(document);
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
    async fn test_csv_loader() {
        // text to represent csv data
        let input = "name,age,city,country
John Doe,25,New York,United States
Jane Smith,32,London,United Kingdom";

        let columns = vec![
            "name".to_string(),
            "age".to_string(),
            "city".to_string(),
            "country".to_string(),
        ];
        let csv_loader = CsvLoader::new(input.as_bytes(), columns);

        let documents = csv_loader.load().await.expect("Failed to load documents");

        assert_eq!(documents.len(), 2);

        let expected1 = "name: John Doe\nage: 25\ncity: New York\ncountry: United States\n";
        assert_eq!(documents[0].metadata.get("row").unwrap(), &Value::from(1));
        assert_eq!(documents[0].page_content, expected1);

        let expected2 = "name: Jane Smith\nage: 32\ncity: London\ncountry: United Kingdom\n";
        assert_eq!(documents[1].metadata.get("row").unwrap(), &Value::from(2));
        assert_eq!(documents[1].page_content, expected2);
    }

    #[tokio::test]
    async fn test_csv_load_from_path() {
        let path = "./src/document_loaders/test_data/test.csv";
        let columns = vec![
            "name".to_string(),
            "age".to_string(),
            "city".to_string(),
            "country".to_string(),
        ];
        let csv_loader = CsvLoader::from_path(path, columns)
            .await
            .expect("Failed to create csv loader");

        let documents = csv_loader.load().await.expect("Failed to load documents");

        assert_eq!(documents.len(), 20);

        let expected1 = "name: John Doe\nage: 25\ncity: New York\ncountry: United States\n";
        assert_eq!(documents[0].metadata.get("row").unwrap(), &Value::from(1));
        assert_eq!(documents[0].page_content, expected1);

        let expected2 = "name: Jane Smith\nage: 32\ncity: London\ncountry: United Kingdom\n";
        assert_eq!(documents[1].metadata.get("row").unwrap(), &Value::from(2));
        assert_eq!(documents[1].page_content, expected2);
    }
}
