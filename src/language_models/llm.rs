use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::schemas::{Message, StreamData};

use super::{options::CallOptions, GenerateResult, LLMError};

#[async_trait]
pub trait LLM: Sync + Send + LLMClone {
    /// This is usefull when you want to create a chain and override
    /// LLM options
    fn add_options(&mut self, options: CallOptions);

    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError>;
    async fn invoke(&self, prompt: &str) -> Result<String, LLMError> {
        self.generate(&[Message::new_human_message(prompt)])
            .await
            .map(|res| res.generation)
    }
    async fn stream(
        &self,
        _messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError>;

    //This is usefull when using non chat models
    fn messages_to_string(&self, messages: &[Message]) -> String {
        messages
            .iter()
            .map(|m| format!("{:?}: {}", m.message_type, m.content))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

pub trait LLMClone {
    fn clone_box(&self) -> Box<dyn LLM>;
}

impl<T> LLMClone for T
where
    T: 'static + LLM + Clone,
{
    fn clone_box(&self) -> Box<dyn LLM> {
        Box::new(self.clone())
    }
}

impl<L> From<L> for Box<dyn LLM>
where
    L: 'static + LLM,
{
    fn from(llm: L) -> Self {
        Box::new(llm)
    }
}
