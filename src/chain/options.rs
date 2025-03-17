use futures::Future;

use crate::{language_models::options::CallOptions, schemas::StreamingFunc};

pub struct ChainCallOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stop_words: Option<Vec<String>>,
    pub streaming_func: Option<Box<StreamingFunc>>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: Option<i64>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub repetition_penalty: Option<f32>,
}

impl Default for ChainCallOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl ChainCallOptions {
    pub fn new() -> Self {
        Self {
            max_tokens: None,
            temperature: None,
            stop_words: None,
            streaming_func: None,
            top_k: None,
            top_p: None,
            seed: None,
            min_length: None,
            max_length: None,
            repetition_penalty: None,
        }
    }

    pub fn to_llm_options(options: ChainCallOptions) -> CallOptions {
        let mut llm_option = CallOptions::new();
        if let Some(max_tokens) = options.max_tokens {
            llm_option = llm_option.with_max_tokens(max_tokens);
        }
        if let Some(temperature) = options.temperature {
            llm_option = llm_option.with_temperature(temperature);
        }
        if let Some(stop_words) = options.stop_words {
            llm_option = llm_option.with_stop_words(stop_words);
        }
        if let Some(top_k) = options.top_k {
            llm_option = llm_option.with_top_k(top_k);
        }
        if let Some(top_p) = options.top_p {
            llm_option = llm_option.with_top_p(top_p);
        }
        if let Some(seed) = options.seed {
            llm_option = llm_option.with_seed(seed);
        }
        if let Some(repetition_penalty) = options.repetition_penalty {
            llm_option = llm_option.with_repetition_penalty(repetition_penalty);
        }

        if let Some(streaming_func) = options.streaming_func {
            llm_option = llm_option.with_streaming_func(streaming_func)
        }
        llm_option
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_stop_words(mut self, stop_words: Vec<String>) -> Self {
        self.stop_words = Some(stop_words);
        self
    }

    //TODO:Check if this should be a &str instead of a String
    pub fn with_streaming_func<F, Fut>(mut self, mut func: F) -> Self
    where
        F: FnMut(String) -> Fut + Send + 'static,
        Fut: Future<Output = Result<(), ()>> + Send + 'static,
    {
        self.streaming_func = Some(Box::new(move |s: String| Box::pin(func(s))));
        self
    }

    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_min_length(mut self, min_length: usize) -> Self {
        self.min_length = Some(min_length);
        self
    }

    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = Some(repetition_penalty);
        self
    }
}
