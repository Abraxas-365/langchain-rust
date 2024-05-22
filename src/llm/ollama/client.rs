use crate::{
    language_models::{llm::LLM, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, MessageType, StreamData},
};
use async_trait::async_trait;
use futures::{Stream, TryStreamExt};
use ollama_rs::{
    error::OllamaError,
    generation::{
        chat::{request::ChatMessageRequest, ChatMessage, MessageRole},
        options::GenerationOptions,
    },
    Ollama as OllamaClient,
};
use std::pin::Pin;
use std::sync::Arc;

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

        let stream = result
            .map_ok(|data| StreamData {
                // value: serde_json::to_value(data).map_err(LLMError::from),
                value: serde_json::Value::default(),
                content: data.message.unwrap().content,
            })
            .map_err(|_| OllamaError::from("Ollama stream error".to_string()).into());

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio_stream::StreamExt;

    #[tokio::test]
    #[ignore]
    async fn test_generate() {
        let ollama = Ollama::default().with_model("llama3");
        let response = ollama
            .invoke("Explain what Rayleigh scattering is.")
            .await
            .unwrap();
        println!("{}", response);
    }

    #[tokio::test]
    #[ignore]
    async fn test_stream() {
        let ollama = Ollama::default().with_model("llama3");

        let message = Message::new_human_message("Why does water boil at 100 degrees?");
        let mut stream = ollama.stream(&vec![message]).await.unwrap();
        let mut stdout = tokio::io::stdout();
        while let Some(res) = stream.next().await {
            let data = res.unwrap();
            stdout.write(data.content.as_bytes()).await.unwrap();
        }
        stdout.flush().await.unwrap();
    }
}
