use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::{
    chain::llm_chain::LLMChain,
    language_models::{llm::LLM, GenerateResult},
    prompt::{HumanMessagePromptTemplate, PromptArgs},
    schemas::{
        memory::{BaseMemory, SimpleMemory},
        messages::Message,
    },
    template_fstring,
};

use self::prompt::DEFAULT_TEMPLATE;

use super::{chain_trait::Chain, options::ChainCallOptions};

mod prompt;
pub struct ConversationalChain {
    llm: Box<dyn Chain>,
    memory: Arc<Mutex<dyn BaseMemory>>,
}

impl ConversationalChain {
    pub fn new<L: LLM + 'static>(llm: L) -> Self {
        let prompt = HumanMessagePromptTemplate::new(template_fstring!(
            DEFAULT_TEMPLATE,
            "history",
            "input"
        ));
        let llm_chain = LLMChain::new(prompt, llm);
        Self {
            llm: Box::new(llm_chain),
            memory: Arc::new(Mutex::new(SimpleMemory::new())),
        }
    }

    pub fn new_with_options<L: LLM + 'static>(llm: L, options: ChainCallOptions) -> Self {
        let prompt = HumanMessagePromptTemplate::new(template_fstring!(
            DEFAULT_TEMPLATE,
            "history",
            "input"
        ));
        let llm_chain = LLMChain::new(prompt, llm).with_options(options);
        Self {
            llm: Box::new(llm_chain),
            memory: Arc::new(Mutex::new(SimpleMemory::new())),
        }
    }

    pub fn with_memory(mut self, memory: Arc<Mutex<dyn BaseMemory>>) -> Self {
        self.memory = memory;
        self
    }
}

#[async_trait]
impl Chain for ConversationalChain {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, Box<dyn Error>> {
        let mut memory = self.memory.lock().await;
        let mut input_variables = input_variables;
        input_variables.insert("history".to_string(), memory.to_string());
        let result = self.llm.call(input_variables).await?;
        memory.add_message(Message::new_ai_message(&result.generation));
        Ok(result)
    }

    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>> {
        let mut memory = self.memory.lock().await;
        let mut input_variables = input_variables;
        input_variables.insert("history".to_string(), memory.to_string());
        let result = self.llm.invoke(input_variables).await?;
        memory.add_message(Message::new_ai_message(&result));
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        llm::openai::{OpenAI, OpenAIModel},
        prompt_args,
    };

    use super::*;

    #[tokio::test]
    async fn test_invoke_conversational() {
        let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
        let chain = ConversationalChain::new(llm);

        let input_variables = prompt_args! {
            "input" => "Soy de peru",
        };
        match chain.invoke(input_variables).await {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }

        let input_variables = prompt_args! {
            "input" => "Cuales son platos tipicos de mi pais",
        };
        match chain.invoke(input_variables).await {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
    }
}
