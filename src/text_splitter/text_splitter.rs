use std::error::Error;

pub trait TextSplitter {
    fn split_text(text: &str) -> Result<Vec<String>, Box<dyn Error>>;
}
