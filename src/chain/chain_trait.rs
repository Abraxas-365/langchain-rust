use std::error::Error;

use async_trait::async_trait;

use crate::{
    language_models::{options::CallOptions, GenerateResult},
    prompt::PromptArgs,
};

use super::options::ChainCallOptions;

#[async_trait]
pub trait Chain: Sync + Send {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, Box<dyn Error>>;
    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>>;
    fn get_options(&self, options: ChainCallOptions) -> CallOptions {
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
        if let Some(min_length) = options.min_length {
            llm_option = llm_option.with_min_length(min_length);
        }
        if let Some(max_length) = options.max_length {
            llm_option = llm_option.with_max_length(max_length);
        }
        if let Some(repetition_penalty) = options.repetition_penalty {
            llm_option = llm_option.with_repetition_penalty(repetition_penalty);
        }

        if let Some(streaming_func) = options.streaming_func {
            llm_option = llm_option.with_streaming_func(streaming_func)
        }
        llm_option
    }
}
