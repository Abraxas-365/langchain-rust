use std::error::Error;

use tiktoken_rs::{get_bpe_from_model, get_bpe_from_tokenizer, tokenizer::Tokenizer, CoreBPE};

use super::{SplitterOptions, TextSplitter};

pub struct TokenSplitter {
    chunk_size: usize,
    model_name: String,
    encoding_name: String,
    trim_chunks: bool,
}

impl Default for TokenSplitter {
    fn default() -> Self {
        TokenSplitter::new(SplitterOptions::default())
    }
}

impl TokenSplitter {
    pub fn new(options: SplitterOptions) -> TokenSplitter {
        TokenSplitter {
            chunk_size: options.chunk_size,
            model_name: options.model_name,
            encoding_name: options.encoding_name,
            trim_chunks: options.trim_chunks,
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

    fn split(&self, text: &str, tokenizer: CoreBPE) -> Vec<String> {
        let splitter =
            text_splitter::TextSplitter::new(tokenizer).with_trim_chunks(self.trim_chunks);
        splitter
            .chunks(text, self.chunk_size)
            .map(|x| x.to_string())
            .collect()
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
