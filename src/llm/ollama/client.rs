use std::sync::Arc;

use crate::{
    language_models::{llm::LLM, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, MessageType, StreamData},
};
use async_trait::async_trait;
use futures::Stream;
use ollama_rs::{
    error::OllamaError,
    generation::{
        chat::{request::ChatMessageRequest, ChatMessage, MessageRole},
        options::GenerationOptions,
    },
    Ollama as OllamaClient,
};

use std::pin::Pin;

#[derive(Debug, Clone)]
pub struct Ollama {
    pub(crate) client: Arc<OllamaClient>,
    pub(crate) model: String,
    pub(crate) options: Option<GenerationOptions>,
}

/// [llama3](https://ollama.com/library/llama3) is a 8B parameters, 4.7GB model.
const DEFAULT_MODEL: &str = "llama3";

impl Ollama {
    pub fn new<S: Into<String>>(
        client: Arc<OllamaClient>,
        model: S,
        options: Option<GenerationOptions>,
    ) -> Self {
        Ollama {
            client,
            model: model.into(),
            options,
        }
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_options(mut self, options: GenerationOptions) -> Self {
        self.options = Some(options);
        self
    }

    fn generate_request(&self, messages: &[Message]) -> ChatMessageRequest {
        let mapped_messages = messages.iter().map(|message| message.into()).collect();
        ChatMessageRequest::new(self.model.clone(), mapped_messages)
    }
}

impl From<&Message> for ChatMessage {
    fn from(message: &Message) -> Self {
        ChatMessage {
            content: message.content.clone(),
            images: None,
            role: message.message_type.clone().into(),
        }
    }
}

impl From<MessageType> for MessageRole {
    fn from(message_type: MessageType) -> Self {
        match message_type {
            MessageType::AIMessage => MessageRole::Assistant,
            MessageType::ToolMessage => MessageRole::Assistant,
            MessageType::SystemMessage => MessageRole::System,
            MessageType::HumanMessage => MessageRole::User,
        }
    }
}

impl Default for Ollama {
    fn default() -> Self {
        let client = Arc::new(OllamaClient::default());
        Ollama::new(client, String::from(DEFAULT_MODEL), None)
    }
}

#[async_trait]
impl LLM for Ollama {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let request = self.generate_request(messages);
        let result = self.client.send_chat_messages(request).await?;

        let generation = match result.message {
            Some(message) => message.content,
            None => return Err(OllamaError::from("No message in response".to_string()).into()),
        };

        let tokens = result.final_data.map(|final_data| {
            let prompt_tokens = final_data.prompt_eval_count as u32;
            let completion_tokens = final_data.eval_count as u32;
            TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            }
        });

        Ok(GenerateResult { tokens, generation })
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let request = self.generate_request(messages);
        let result = self.client.send_chat_messages_stream(request).await?;

        // Err(OllamaError::from("Stream error".to_string()).into())
        // let stream = result
        //     .map(|data| {
        //         data.map(|d| StreamData {
        //             content: "???".to_string(),
        //             value: serde_json::Value::default(),
        //         })
        //     })
        //     .collect();

        todo!("not sure about how to map the stream")
    }
}
