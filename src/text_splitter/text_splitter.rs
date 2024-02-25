use std::error::Error;

pub trait TextSplitter {
    fn split_text(&self, text: &str) -> Result<Vec<String>, Box<dyn Error>>;
}
