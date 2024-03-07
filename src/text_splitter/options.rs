// Options is a struct that contains options for a text splitter.
pub struct SplitterOptions {
    pub chunk_size: usize,
    pub model_name: String,
    pub encoding_name: String,
    pub trim_chunks: bool,
}

impl Default for SplitterOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl SplitterOptions {
    pub fn new() -> Self {
        SplitterOptions {
            chunk_size: 512,
            model_name: String::from("gpt-3.5-turbo"),
            encoding_name: String::from("cl100k_base"),
            trim_chunks: false,
        }
    }
}

// Builder pattern for Options struct
impl SplitterOptions {
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    pub fn with_model_name(mut self, model_name: &str) -> Self {
        self.model_name = String::from(model_name);
        self
    }

    pub fn with_encoding_name(mut self, encoding_name: &str) -> Self {
        self.encoding_name = String::from(encoding_name);
        self
    }

    pub fn with_trim_chunks(mut self, trim_chunks: bool) -> Self {
        self.trim_chunks = trim_chunks;
        self
    }
}
