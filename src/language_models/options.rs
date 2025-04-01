use async_openai::types::{ChatCompletionTool, ChatCompletionToolChoiceOption, ResponseFormat};
use futures::Future;
use std::{fmt, pin::Pin, sync::Arc};
use tokio::sync::Mutex;

use crate::schemas::StreamingFunc;

#[derive(Clone, Default)]
pub struct StreamOption {
    pub streaming_func: Option<Arc<Mutex<StreamingFunc>>>,
    pub include_usage: bool,
}

impl fmt::Debug for StreamOption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamOption")
            .field("streaming_func", &self.streaming_func.is_some())
            .field("include_usage", &self.include_usage)
            .finish()
    }
}

impl StreamOption {
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

    pub fn with_stream_usage(mut self, stream_usage: bool) -> Self {
        self.include_usage = stream_usage;
        self
    }
}

#[derive(Clone, Debug)]
pub struct CallOptions {
    pub candidate_count: Option<usize>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stop_words: Option<Vec<String>>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub seed: Option<i64>,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub n: Option<u8>,
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub tools: Option<Vec<ChatCompletionTool>>,
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    pub response_format: Option<ResponseFormat>,
    pub stream_option: Option<StreamOption>,
    pub system_is_assistant: bool,
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
            top_k: None,
            top_p: None,
            seed: None,
            min_length: None,
            max_length: None,
            n: None,
            repetition_penalty: None,
            frequency_penalty: None,
            presence_penalty: None,
            tools: None,
            tool_choice: None,
            response_format: None,
            stream_option: None,
            system_is_assistant: false,
        }
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

    pub fn with_n(mut self, n: u8) -> Self {
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

    pub fn with_tools(mut self, tools: Vec<ChatCompletionTool>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: ChatCompletionToolChoiceOption) -> Self {
        self.tool_choice = Some(tool_choice);
        self
    }

    pub fn with_response_format(mut self, response_format: ResponseFormat) -> Self {
        self.response_format = Some(response_format);
        self
    }

    pub fn with_stream(mut self, stream: StreamOption) -> Self {
        self.stream_option = Some(stream);
        self
    }

    pub fn with_system_is_assistant(mut self, system_is_assistant: bool) -> Self {
        self.system_is_assistant = system_is_assistant;
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
        self.tool_choice = incoming_options.tool_choice.or(self.tool_choice.clone());
        self.response_format = incoming_options
            .response_format
            .or(self.response_format.clone());

        // For `Vec<String>`, merge if both are Some; prefer incoming if only incoming is Some
        if let Some(mut new_stop_words) = incoming_options.stop_words {
            if let Some(existing_stop_words) = &mut self.stop_words {
                existing_stop_words.append(&mut new_stop_words);
            } else {
                self.stop_words = Some(new_stop_words);
            }
        }

        // For `Vec<FunctionDefinition>`, similar logic to `Vec<String>`
        if let Some(mut incoming_functions) = incoming_options.tools {
            if let Some(existing_functions) = &mut self.tools {
                existing_functions.append(&mut incoming_functions);
            } else {
                self.tools = Some(incoming_functions);
            }
        }

        if let Some(stream) = incoming_options.stream_option {
            self.stream_option = Some(stream);
        }

        self.system_is_assistant = self.system_is_assistant || incoming_options.system_is_assistant;
    }
}
