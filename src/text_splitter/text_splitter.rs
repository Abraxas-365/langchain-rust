use std::{collections::HashMap, error::Error};

use serde_json::Value;

use crate::schemas::Document;

pub trait TextSplitter {
    fn split_text(&self, text: &str) -> Result<Vec<String>, Box<dyn Error>>;

    fn split_documents(&self, documents: &[Document]) -> Result<Vec<Document>, Box<dyn Error>> {
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
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        let mut metadatas = metadatas.to_vec();
        if metadatas.is_empty() {
            metadatas = vec![HashMap::new(); text.len()];
        }

        if text.len() != metadatas.len() {
            return Err(Box::from("Mismatch metadatas and text"));
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

pub(crate) fn join_documents(docs: &[String], separator: &str) -> Option<String> {
    let text = docs.join(separator);
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
}

pub(crate) fn merge_splits(
    splits: &[String],
    separator: &str,
    chunk_size: usize,
    chunk_overlap: usize,
    len_func: fn(&str) -> usize,
) -> Vec<String> {
    let mut docs: Vec<String> = Vec::new();
    let mut current_doc: Vec<String> = Vec::new();
    let mut total = 0;
    for split in splits {
        let mut total_with_split = total + len_func(split);
        if !current_doc.is_empty() {
            total_with_split += len_func(separator);
        }
        if total_with_split > chunk_size && !current_doc.is_empty() {
            let doc = join_documents(&current_doc, separator);
            if let Some(doc) = doc {
                docs.push(doc);
            }
            while should_pop(
                chunk_overlap,
                chunk_size,
                total,
                len_func(split),
                len_func(separator),
                current_doc.len(),
            ) {
                total -= len_func(&current_doc[0]);
                if current_doc.len() > 1 {
                    total -= len_func(separator);
                }
                current_doc.remove(0);
            }
        }
        current_doc.push(split.to_string());
        total += len_func(split);
        if current_doc.len() > 1 {
            total += len_func(separator);
        }
    }
    let doc = join_documents(&current_doc, separator);
    if let Some(doc) = doc {
        docs.push(doc);
    }
    docs
}

pub(crate) fn should_pop(
    chunk_overlap: usize,
    chunk_size: usize,
    total: usize,
    split_len: usize,
    separator_len: usize,
    current_doc_len: usize,
) -> bool {
    let docs_needed_to_add_sep = 2;
    let separator_len = match current_doc_len < docs_needed_to_add_sep {
        true => 0,
        false => separator_len,
    };

    current_doc_len > 0
        && (total > chunk_overlap || (total + split_len + separator_len > chunk_size && total > 0))
}

// func shouldPop(chunkOverlap, chunkSize, total, splitLen, separatorLen, currentDocLen int) bool {
// 	docsNeededToAddSep := 2
// 	if currentDocLen < docsNeededToAddSep {
// 		separatorLen = 0
// 	}
//
// 	return currentDocLen > 0 && (total > chunkOverlap || (total+splitLen+separatorLen > chunkSize && total > 0))
// }
