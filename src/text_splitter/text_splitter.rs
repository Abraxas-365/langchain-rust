use std::collections::HashMap;

use serde_json::Value;

use crate::schemas::Document;

use super::TextSplitterError;

pub trait TextSplitter: Send + Sync {
    fn split_text(&self, text: &str) -> Result<Vec<String>, TextSplitterError>;

    fn split_documents(&self, documents: &[Document]) -> Result<Vec<Document>, TextSplitterError> {
        let mut texts: Vec<String> = Vec::new();
        let mut metadatas: Vec<HashMap<String, Value>> = Vec::new();
        documents.iter().for_each(|d| {
            texts.push(d.page_content.clone());
            metadatas.push(d.metadata.clone());
        });

        self.create_documents(&texts, &metadatas)
    }

    fn create_documents(
        &self,
        text: &[String],
        metadatas: &[HashMap<String, Value>],
    ) -> Result<Vec<Document>, TextSplitterError> {
        let mut metadatas = metadatas.to_vec();
        if metadatas.is_empty() {
            metadatas = vec![HashMap::new(); text.len()];
        }

        if text.len() != metadatas.len() {
            return Err(TextSplitterError::MetadataTextMismatch);
        }

        let mut documents: Vec<Document> = Vec::new();
        for i in 0..text.len() {
            let chunks = self.split_text(&text[i])?;
            for chunk in chunks {
                let document = Document::new(chunk).with_metadata(metadatas[i].clone());
                documents.push(document);
            }
        }

        Ok(documents)
    }
}
