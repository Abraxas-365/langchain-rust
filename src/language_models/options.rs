use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone, Copy, Debug)]
pub enum FunctionCallBehavior {
    None,
    Auto,
}

#[derive(Clone, Debug)]
pub struct FunctionDefinition {
    name: String,
    description: String,
    parameters: HashMap<String, String>,
}

#[derive(Clone)]
pub struct CallOptions {
    candidate_count: Option<usize>,
    max_tokens: Option<u16>,
    temperature: Option<f32>,
    stop_words: Option<Vec<String>>,
    streaming_func: Option<Arc<Mutex<dyn FnMut(Vec<u8>) -> Result<(), ()> + Send>>>,
    top_k: Option<usize>,
    top_p: Option<f32>,
    seed: Option<usize>,
    min_length: Option<usize>,
    max_length: Option<usize>,
    n: Option<usize>,
    repetition_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    functions: Option<Vec<FunctionDefinition>>,
    function_call_behavior: Option<FunctionCallBehavior>,
}

impl CallOptions {
    fn new() -> Self {
        CallOptions {
            candidate_count: None,
            max_tokens: None,
            temperature: None,
            stop_words: None,
            streaming_func: None,
            top_k: None,
            top_p: None,
            seed: None,
            min_length: None,
            max_length: None,
            n: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            functions: None,
            function_call_behavior: None,
        }
    }

    // Refactored "with" functions as methods of CallOptions
    fn with_max_tokens(&mut self, max_tokens: u16) -> &mut Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    fn with_candidate_count(&mut self, candidate_count: usize) -> &mut Self {
        self.candidate_count = Some(candidate_count);
        self
    }

    fn with_temperature(&mut self, temperature: f32) -> &mut Self {
        self.temperature = Some(temperature);
        self
    }

    fn with_stop_words(&mut self, stop_words: Vec<String>) -> &mut Self {
        self.stop_words = Some(stop_words);
        self
    }

    fn with_streaming_func<F>(&mut self, func: F) -> &mut Self
    where
        F: FnMut(Vec<u8>) -> Result<(), ()> + Send + 'static,
    {
        let func = Arc::new(Mutex::new(func));
        self.streaming_func = Some(func);
        self
    }

    fn with_top_k(&mut self, top_k: usize) -> &mut Self {
        self.top_k = Some(top_k);
        self
    }

    fn with_top_p(&mut self, top_p: f32) -> &mut Self {
        self.top_p = Some(top_p);
        self
    }

    fn with_seed(&mut self, seed: usize) -> &mut Self {
        self.seed = Some(seed);
        self
    }

    fn with_min_length(&mut self, min_length: usize) -> &mut Self {
        self.min_length = Some(min_length);
        self
    }

    fn with_max_length(&mut self, max_length: usize) -> &mut Self {
        self.max_length = Some(max_length);
        self
    }

    fn with_n(&mut self, n: usize) -> &mut Self {
        self.n = Some(n);
        self
    }

    fn with_repetition_penalty(&mut self, repetition_penalty: f32) -> &mut Self {
        self.repetition_penalty = Some(repetition_penalty);
        self
    }

    fn with_frequency_penalty(&mut self, frequency_penalty: f32) -> &mut Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    fn with_presence_penalty(&mut self, presence_penalty: f32) -> &mut Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    fn with_functions(&mut self, functions: Vec<FunctionDefinition>) -> &mut Self {
        self.functions = Some(functions);
        self
    }

    fn with_function_call_behavior(&mut self, behavior: FunctionCallBehavior) -> &mut Self {
        self.function_call_behavior = Some(behavior);
        self
    }
}
