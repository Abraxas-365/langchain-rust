use std::{error::Error, sync::Arc};

use async_trait::async_trait;

use crate::{
    language_models::{llm::LLM, options::CallOptions, GenerateResult},
    prompt::{FormatPrompter, PromptArgs},
    schemas::memory::BaseMemory,
};

use super::{chain_trait::Chain, options::ChainCallOptions};

pub struct LLMChain<P, L>
where
    P: FormatPrompter,
    L: LLM,
{
    prompt: P,
    llm: L,
    memory: Option<Arc<dyn BaseMemory>>,
}

impl<P, L> LLMChain<P, L>
where
    P: FormatPrompter,
    L: LLM,
{
    pub fn new(prompt: P, llm: L) -> Self {
        Self {
            prompt,
            llm,
            memory: None,
        }
    }

    pub fn with_memory(mut self, memory: Arc<dyn BaseMemory>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_options(mut self, options: ChainCallOptions) -> Self {
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

        self.llm.with_options(llm_option);
        self
    }
}

#[async_trait]
impl<P, L> Chain for LLMChain<P, L>
where
    P: FormatPrompter + Send + Sync,
    L: LLM + Send + Sync,
{
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, Box<dyn Error>> {
        let prompt = self.prompt.format_prompt(input_variables)?;
        self.llm.generate(&prompt.to_chat_messages()).await
    }

    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>> {
        let prompt = self.prompt.format_prompt(input_variables)?;
        self.llm.invoke(&prompt.to_string()).await
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        llm::openai::{OpenAI, OpenAIModel},
        message_formatter, messages_placeholder,
        prompt::{AIMessagePromptTemplate, HumanMessagePromptTemplate, MessageOrTemplate},
        prompt_args,
        schemas::messages::Message,
        template_fstring,
    };

    use super::*;
    use futures::lock::Mutex;

    #[tokio::test]
    async fn test_invoke_chain() {
        let human_msg = Message::new_human_message("Hello from user");

        // Create an AI message prompt template
        let human_message_prompt = HumanMessagePromptTemplate::new(template_fstring!(
            "Mi nombre es: {nombre} ",
            "nombre",
        ));

        let message_complete = Arc::new(Mutex::new(String::new()));

        // Define the streaming function
        // This function will append the content received from the stream to `message_complete`
        let streaming_func = {
            let message_complete = message_complete.clone();
            move |content: String| {
                let message_complete = message_complete.clone();
                async move {
                    let mut message_complete_lock = message_complete.lock().await;
                    println!("Content: {:?}", content);
                    message_complete_lock.push_str(&content);
                    Ok(())
                }
            }
        };
        // Use the `message_formatter` macro to construct the formatter
        let formatter = message_formatter![MessageOrTemplate::Template(human_message_prompt),];

        let options = ChainCallOptions::default().with_streaming_func(streaming_func);
        let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
        let chain = LLMChain::new(formatter, llm).with_options(options);

        let input_variables = prompt_args! {
            "nombre" => "luis",

        };
        match chain.invoke(input_variables).await {
            Ok(result) => {
                println!("Result: {:?}", result);
                println!("Complete message: {:?}", message_complete.lock().await);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
    }
}
