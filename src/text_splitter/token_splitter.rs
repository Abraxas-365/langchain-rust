use std::{collections::HashSet, error::Error};

use tiktoken_rs::{get_bpe_from_model, get_bpe_from_tokenizer, tokenizer::Tokenizer, CoreBPE};

use super::{SplitterOptions, TextSplitter};

pub struct TokenSplitter {
    chunk_size: usize,
    chunk_overlap: usize,
    model_name: String,
    encoding_name: String,
    allowed_special: Vec<String>,
    disallowed_special: Vec<String>,
}

impl TokenSplitter {
    pub fn new(options: SplitterOptions) -> TokenSplitter {
        TokenSplitter {
            chunk_size: options.chunk_size,
            chunk_overlap: options.chunk_overlap,
            model_name: options.model_name,
            encoding_name: options.encoding_name,
            allowed_special: options.allowed_special,
            disallowed_special: options.disallowed_special,
        }
    }

    pub fn get_tokenizer_from_str(&self, s: &str) -> Option<Tokenizer> {
        match s.to_lowercase().as_str() {
            "cl100k_base" => Some(Tokenizer::Cl100kBase),
            "p50k_base" => Some(Tokenizer::P50kBase),
            "r50k_base" => Some(Tokenizer::R50kBase),
            "p50k_edit" => Some(Tokenizer::P50kEdit),
            "gpt2" => Some(Tokenizer::Gpt2),
            _ => None,
        }
    }
}
impl TokenSplitter {
    fn split(&self, text: &str, tokenizer: CoreBPE) -> Vec<String> {
        let mut splits = Vec::new();
        let allowed_special: HashSet<&str> =
            self.allowed_special.iter().map(|s| s.as_str()).collect();
        let input_ids = tokenizer.encode(text, allowed_special);
        let mut start_idx = 0;
        let mut cur_idx = input_ids.len();
        if start_idx + self.chunk_size < cur_idx {
            cur_idx = start_idx + self.chunk_size;
        }
        while start_idx < input_ids.len() {
            let chunk_ids = input_ids[start_idx as usize..cur_idx as usize].to_vec();
            splits.push(tokenizer.decode(chunk_ids).unwrap_or_default());
            start_idx += self.chunk_size - self.chunk_overlap;
            cur_idx = start_idx + self.chunk_size;
            if cur_idx > input_ids.len() {
                cur_idx = input_ids.len();
            }
        }
        splits
    }
}

impl TextSplitter for TokenSplitter {
    fn split_text(&self, text: &str) -> Result<Vec<String>, Box<dyn Error>> {
        let tk = if !self.encoding_name.is_empty() {
            let tokenizer = self
                .get_tokenizer_from_str(&self.encoding_name)
                .ok_or("Tokenizer not found")?;
            let tokenizer = get_bpe_from_tokenizer(tokenizer)?;
            tokenizer
        } else {
            let tokenizer = get_bpe_from_model(&self.model_name)?;
            tokenizer
        };
        let text = self.split(text, tk);
        Ok(text)
    }
}
