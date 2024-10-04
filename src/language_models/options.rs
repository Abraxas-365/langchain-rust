use crate::tools::{Tool, ToolCallBehavior};
use futures::Future;
use std::{pin::Pin, sync::Arc};
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct CallOptions {
    pub mirostat: Option<u8>,
    pub mirostat_eta: Option<f32>,
    pub mirostat_tau: Option<f32>,
    pub num_ctx: Option<u32>,
    pub num_gqa: Option<u32>,
    pub num_gpu: Option<u32>,
    pub num_thread: Option<u32>,
    pub repeat_last_n: Option<i32>,
    pub tfs_z: Option<f32>,
    pub num_predict: Option<i32>,
    pub candidate_count: Option<usize>,
    pub max_tokens: Option<u32>,
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
    pub functions: Option<Vec<Arc<dyn Tool>>>,
    pub function_call_behavior: Option<ToolCallBehavior>,
}

impl Default for CallOptions {
    fn default() -> Self {
        CallOptions::new()
    }
}
impl CallOptions {
    pub fn new() -> Self {
        CallOptions {
            mirostat: None,
            mirostat_eta: None,
            mirostat_tau: None,
            num_ctx: None,
            num_gqa: None,
            num_gpu: None,
            num_thread: None,
            repeat_last_n: None,
            tfs_z: None,
            num_predict: None,
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

    pub fn with_mirostat(mut self, mirostat: u8) -> Self {
        self.mirostat = Some(mirostat);
        self
    }

    pub fn with_mirostat_eta(mut self, mirostat_eta: f32) -> Self {
        self.mirostat_eta = Some(mirostat_eta);
        self
    }

    pub fn with_mirostat_tau(mut self, mirostat_tau: f32) -> Self {
        self.mirostat_tau = Some(mirostat_tau);
        self
    }

    pub fn with_num_ctx(mut self, num_ctx: u32) -> Self {
        self.num_ctx = Some(num_ctx);
        self
    }

    pub fn with_num_gqa(mut self, num_gqa: u32) -> Self {
        self.num_gqa = Some(num_gqa);
        self
    }

    pub fn with_num_gpu(mut self, num_gpu: u32) -> Self {
        self.num_gpu = Some(num_gpu);
        self
    }

    pub fn with_num_thread(mut self, num_thread: u32) -> Self {
        self.num_thread = Some(num_thread);
        self
    }

    pub fn with_repeat_last_n(mut self, repeat_last_n: i32) -> Self {
        self.repeat_last_n = Some(repeat_last_n);
        self
    }

    pub fn with_tfs_z(mut self, tfs_z: f32) -> Self {
        self.tfs_z = Some(tfs_z);
        self
    }

    pub fn with_num_predict(mut self, num_predict: i32) -> Self {
        self.num_predict = Some(num_predict);
        self
    }

    // Refactored "with" functions as methods of CallOptions
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
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

    pub fn with_functions(mut self, functions: Vec<Arc<dyn Tool>>) -> Self {
        self.functions = Some(functions);
        self
    }

    pub fn with_function_call_behavior(mut self, behavior: ToolCallBehavior) -> Self {
        self.function_call_behavior = Some(behavior);
        self
    }

    pub fn merge_options(&mut self, incoming_options: CallOptions) {
        // For simple scalar types wrapped in Option, prefer incoming option if it is Some
        self.candidate_count = incoming_options.candidate_count.or(self.candidate_count);
        self.max_tokens = incoming_options.max_tokens.or(self.max_tokens);
        self.temperature = incoming_options.temperature.or(self.temperature);
        self.top_k = incoming_options.top_k.or(self.top_k);
        self.top_p = incoming_options.top_p.or(self.top_p);
        self.seed = incoming_options.seed.or(self.seed);
        self.min_length = incoming_options.min_length.or(self.min_length);
        self.max_length = incoming_options.max_length.or(self.max_length);
        self.n = incoming_options.n.or(self.n);
        self.repetition_penalty = incoming_options
            .repetition_penalty
            .or(self.repetition_penalty);
        self.frequency_penalty = incoming_options
            .frequency_penalty
            .or(self.frequency_penalty);
        self.presence_penalty = incoming_options.presence_penalty.or(self.presence_penalty);
        self.function_call_behavior = incoming_options
            .function_call_behavior
            .or(self.function_call_behavior.clone());

        // For `Vec<String>`, merge if both are Some; prefer incoming if only incoming is Some
        if let Some(mut new_stop_words) = incoming_options.stop_words {
            if let Some(existing_stop_words) = &mut self.stop_words {
                existing_stop_words.append(&mut new_stop_words);
            } else {
                self.stop_words = Some(new_stop_words);
            }
        }

        // For `Vec<FunctionDefinition>`, similar logic to `Vec<String>`
        if let Some(mut incoming_functions) = incoming_options.functions {
            if let Some(existing_functions) = &mut self.functions {
                existing_functions.append(&mut incoming_functions);
            } else {
                self.functions = Some(incoming_functions);
            }
        }

        // `streaming_func` requires a judgment call on how you want to handle merging.
        // Here, the incoming option simply replaces the existing one if it's Some.
        self.streaming_func = incoming_options
            .streaming_func
            .or_else(|| self.streaming_func.clone());
    }
}
