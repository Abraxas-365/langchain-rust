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
        let llm_option = self.get_options(options);
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
        Ok(self
            .llm
            .generate(&prompt.to_chat_messages())
            .await?
            .generation)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        llm::openai::{OpenAI, OpenAIModel},
        message_formatter,
        prompt::{HumanMessagePromptTemplate, MessageOrTemplate},
        prompt_args, template_fstring,
    };

    use super::*;
    use futures::lock::Mutex;

    #[tokio::test]
    async fn test_invoke_chain() {
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
        let formatter =
            message_formatter![MessageOrTemplate::Template(human_message_prompt.into()),];

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
