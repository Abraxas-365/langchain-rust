// Options is a struct that contains options for a text splitter.
pub struct SplitterOptions {
    pub chunk_size: i32,
    pub chunk_overlap: i32,
    pub separators: Vec<String>,
    pub len_func: Option<Box<dyn Fn(&str) -> usize>>,
    pub model_name: String,
    pub encoding_name: String,
    pub allowed_special: Vec<String>,
    pub disallowed_special: Vec<String>,
    pub code_blocks: bool,
    pub reference_links: bool,
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
            chunk_overlap: 100,
            separators: vec!["\n\n".into(), "\n".into(), " ".into(), "".into()],
            len_func: Some(Box::new(|s: &str| s.chars().count())),
            model_name: String::from("gpt-3.5-turbo"),
            encoding_name: String::from("cl100k_base"),
            allowed_special: Vec::new(),
            disallowed_special: Vec::from(["all".into()]),
            code_blocks: false,
            reference_links: false,
        }
    }
}

// Builder pattern for Options struct
impl SplitterOptions {
    pub fn with_chunk_size(mut self, chunk_size: i32) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    pub fn with_chunk_overlap(mut self, chunk_overlap: i32) -> Self {
        self.chunk_overlap = chunk_overlap;
        self
    }

    pub fn with_separators(mut self, separators: Vec<&str>) -> Self {
        self.separators = separators.into_iter().map(String::from).collect();
        self
    }

    pub fn with_len_func(mut self, len_func: Box<dyn Fn(&str) -> usize>) -> Self {
        self.len_func = Some(len_func);
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

    pub fn with_allowed_special(mut self, allowed_special: Vec<&str>) -> Self {
        self.allowed_special = allowed_special.into_iter().map(String::from).collect();
        self
    }

    pub fn with_disallowed_special(mut self, disallowed_special: Vec<&str>) -> Self {
        self.disallowed_special = disallowed_special.into_iter().map(String::from).collect();
        self
    }

    pub fn with_code_blocks(mut self, code_blocks: bool) -> Self {
        self.code_blocks = code_blocks;
        self
    }

    pub fn with_reference_links(mut self, reference_links: bool) -> Self {
        self.reference_links = reference_links;
        self
    }
}
