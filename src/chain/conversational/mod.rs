use std::{error::Error, pin::Pin, sync::Arc};

use async_trait::async_trait;
use futures::Stream;
use tokio::sync::Mutex;

use crate::{
    language_models::{llm::LLM, GenerateResult},
    memory::SimpleMemory,
    prompt::{HumanMessagePromptTemplate, PromptArgs},
    schemas::{memory::BaseMemory, messages::Message},
    template_fstring,
};

use self::prompt::DEFAULT_TEMPLATE;

use super::{
    chain_trait::Chain,
    llm_chain::{LLMChain, LLMChainBuilder},
};

pub mod builder;
mod prompt;
pub struct ConversationalChain {
    llm: LLMChain,
    memory: Arc<Mutex<dyn BaseMemory>>,
}

//Conversational Chain is a simple chain to interact with ai as a string of messages
impl ConversationalChain {
    pub fn new<L: LLM + 'static>(llm: L) -> Result<Self, Box<dyn Error>> {
        let prompt = HumanMessagePromptTemplate::new(template_fstring!(
            DEFAULT_TEMPLATE,
            "history",
            "input"
        ));
        let llm_chain = LLMChainBuilder::new().prompt(prompt).llm(llm).build()?; //Using the llm
                                                                                 //chian whitout memroy, because the conversational chain will take care of the history
        Ok(Self {
            llm: llm_chain,
            memory: Arc::new(Mutex::new(SimpleMemory::new())),
        })
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
        input_variables.insert("history".to_string(), memory.to_string().into());
        let result = self.llm.call(input_variables.clone()).await?;
        memory.add_message(Message::new_ai_message(&input_variables["input"]));
        memory.add_message(Message::new_ai_message(&result.generation));
        Ok(result)
    }

    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>> {
        let mut memory = self.memory.lock().await;
        let mut input_variables = input_variables;
        input_variables.insert("history".to_string(), memory.to_string().into());
        let result = self.llm.invoke(input_variables.clone()).await?;
        memory.add_message(Message::new_ai_message(&input_variables["input"]));
        memory.add_message(Message::new_ai_message(&result));
        Ok(result)
    }

    async fn stream(
        &self,
        input_variables: PromptArgs,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<serde_json::Value, Box<dyn Error + Send>>> + Send>>,
        Box<dyn Error>,
    > {
        self.llm.stream(input_variables).await
    }

    fn get_input_keys(&self) -> Vec<String> {
        self.llm.get_input_keys()
    }
}

#[cfg(test)]
mod tests {
    use async_openai::config::OpenAIConfig;

    use crate::{
        chain::conversational::builder::ConversationalChainBuilder,
        language_models::options::CallOptions,
        llm::openai::{OpenAI, OpenAIModel},
        prompt_args,
    };

    use super::*;

    #[tokio::test]
    async fn test_invoke_conversational() {
        let llm = OpenAI::new(OpenAIConfig::default(), CallOptions::default())
            .with_model(OpenAIModel::Gpt35.to_string());
        let chain = ConversationalChainBuilder::new()
            .llm(llm)
            .build()
            .expect("Error building ConversationalChain");

        let input_variables_first = prompt_args! {
            "input" => "Soy de peru",
        };
        // Execute the first `chain.invoke` and assert that it should succeed
        let result_first = chain.invoke(input_variables_first).await;
        assert!(
            result_first.is_ok(),
            "Error invoking LLMChain: {:?}",
            result_first.err()
        );

        // Optionally, if you want to print the successful result, you can do so like this:
        if let Ok(result) = result_first {
            println!("Result: {:?}", result);
        }

        let input_variables_second = prompt_args! {
            "input" => "Cuales son platos tipicos de mi pais",
        };
        // Execute the second `chain.invoke` and assert that it should succeed
        let result_second = chain.invoke(input_variables_second).await;
        assert!(
            result_second.is_ok(),
            "Error invoking LLMChain: {:?}",
            result_second.err()
        );

        // Optionally, if you want to print the successful result, you can do so like this:
        if let Ok(result) = result_second {
            println!("Result: {:?}", result);
        }
    }
}
