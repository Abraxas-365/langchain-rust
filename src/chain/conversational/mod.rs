use std::{pin::Pin, sync::Arc};

use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;
use futures_util::{pin_mut, StreamExt};
use tokio::sync::Mutex;

use crate::{
    language_models::GenerateResult,
    prompt::PromptArgs,
    prompt_args,
    schemas::{memory::BaseMemory, messages::Message, StreamData},
};

const DEFAULT_INPUT_VARIABLE: &str = "input";

use super::{chain_trait::Chain, llm_chain::LLMChain, ChainError};

pub mod builder;
mod prompt;

///This is only usefull when you dont modify the original prompt
pub struct ConversationalChainPromptBuilder {
    input: String,
}

impl ConversationalChainPromptBuilder {
    pub fn new() -> Self {
        Self {
            input: "".to_string(),
        }
    }

    pub fn input<S: Into<String>>(mut self, input: S) -> Self {
        self.input = input.into();
        self
    }

    pub fn build(self) -> PromptArgs {
        prompt_args! {
            DEFAULT_INPUT_VARIABLE => self.input,
        }
    }
}

pub struct ConversationalChain {
    llm: LLMChain,
    input_key: String,
    pub memory: Arc<Mutex<dyn BaseMemory>>,
}

//Conversational Chain is a simple chain to interact with ai as a string of messages
impl ConversationalChain {
    pub fn prompt_builder(&self) -> ConversationalChainPromptBuilder {
        ConversationalChainPromptBuilder::new()
    }
}

#[async_trait]
impl Chain for ConversationalChain {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        let input_variable = &input_variables
            .get(&self.input_key)
            .ok_or(ChainError::MissingInputVariable(self.input_key.clone()))?;
        let human_message = Message::new_human_message(input_variable);

        let history = {
            let memory = self.memory.lock().await;
            memory.to_string()
        };
        let mut input_variables = input_variables;
        input_variables.insert("history".to_string(), history.into());
        let result = self.llm.call(input_variables.clone()).await?;

        let mut memory = self.memory.lock().await;
        memory.add_message(human_message);
        memory.add_message(Message::new_ai_message(&result.generation));
        Ok(result)
    }

    async fn stream<'self_life>(
        &'self_life self,
        input_variables: PromptArgs,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, ChainError>> + Send + 'self_life>>, ChainError>
    {
        let input_variable = &input_variables
            .get(&self.input_key)
            .ok_or(ChainError::MissingInputVariable(self.input_key.clone()))?;
        let human_message = Message::new_human_message(input_variable);

        let history = {
            let memory = self.memory.lock().await;
            memory.to_string()
        };

        let mut input_variables = input_variables;
        input_variables.insert("history".to_string(), history.into());

        let complete_ai_message = Arc::new(Mutex::new(String::new()));
        let complete_ai_message_clone = complete_ai_message.clone();

        let memory = self.memory.clone();

        let stream = self.llm.stream(input_variables).await?;
        let output_stream = stream! {
            pin_mut!(stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(data) => {
                        let mut complete_ai_message_clone =
                            complete_ai_message_clone.lock().await;
                        complete_ai_message_clone.push_str(&data.content);

                        yield Ok(data);
                    },
                    Err(e) => {
                        yield Err(e);
                    }
                }
            }

            let mut memory = memory.lock().await;
            memory.add_message(human_message);
            memory.add_message(Message::new_ai_message(&complete_ai_message.lock().await));
        };

        Ok(Box::pin(output_stream))
    }

    fn get_input_keys(&self) -> Vec<String> {
        vec![self.input_key.clone()]
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        chain::conversational::builder::ConversationalChainBuilder,
        llm::openai::{OpenAI, OpenAIModel},
        prompt_args,
    };

    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_invoke_conversational() {
        let llm = OpenAI::default().with_model(OpenAIModel::Gpt35.to_string());
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
