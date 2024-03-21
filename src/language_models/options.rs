use futures::Future;
use serde_json::Value;
use std::{ops::Deref, pin::Pin, sync::Arc};
use tokio::sync::Mutex;

use crate::tools::Tool;

#[derive(Clone, Copy, Debug)]
pub enum FunctionCallBehavior {
    None,
    Auto,
}

#[derive(Clone, Debug)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl FunctionDefinition {
    pub fn new(name: &str, description: &str, parameters: Value) -> Self {
        FunctionDefinition {
            name: name.to_string(),
            description: description.to_string(),
            parameters,
        }
    }

    /// Generic function that can be used with both Arc<Tool>, Box<Tool>, and direct references
    pub fn from_langchain_tool<T>(tool: &T) -> FunctionDefinition
    where
        T: Deref<Target = dyn Tool + Send + Sync> + ?Sized,
    {
        FunctionDefinition {
            name: tool.name(),
            description: tool.description(),
            parameters: tool.parameters(),
        }
    }
}

#[derive(Clone)]
pub struct CallOptions {
    pub candidate_count: Option<usize>,
    pub max_tokens: Option<u16>,
    pub temperature: Option<f32>,
    pub stop_words: Option<Vec<String>>,
    pub streaming_func: Option<
        Arc<
            Mutex<dyn FnMut(String) -> Pin<Box<dyn Future<Output = Result<(), ()>> + Send>> + Send>,
        >,
    >,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: Option<usize>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub n: Option<usize>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub functions: Option<Vec<FunctionDefinition>>,
    pub function_call_behavior: Option<FunctionCallBehavior>,
}

impl Default for CallOptions {
    fn default() -> Self {
        CallOptions::new()
    }
}
impl CallOptions {
    pub fn new() -> Self {
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
    pub fn with_max_tokens(mut self, max_tokens: u16) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_candidate_count(mut self, candidate_count: usize) -> Self {
        self.candidate_count = Some(candidate_count);
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
        let func = Arc::new(Mutex::new(
            move |s: String| -> Pin<Box<dyn Future<Output = Result<(), ()>> + Send>> {
                Box::pin(func(s))
            },
        ));

        self.streaming_func = Some(func);
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

    pub fn with_seed(mut self, seed: usize) -> Self {
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

    pub fn with_n(mut self, n: usize) -> Self {
        self.n = Some(n);
        self
    }

    pub fn with_repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = Some(repetition_penalty);
        self
    }

    pub fn with_frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    pub fn with_presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    pub fn with_functions(mut self, functions: Vec<FunctionDefinition>) -> Self {
        self.functions = Some(functions);
        self
    }

    pub fn with_function_call_behavior(mut self, behavior: FunctionCallBehavior) -> Self {
        self.function_call_behavior = Some(behavior);
        self
    }
}
