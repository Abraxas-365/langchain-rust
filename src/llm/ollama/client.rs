use crate::{
    language_models::{llm::LLM, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, MessageType, StreamData},
};
use async_trait::async_trait;
use futures::Stream;
use ollama_rs::generation::{chat::ChatMessageFinalResponseData, images::Image};
pub use ollama_rs::{
    error::OllamaError,
    generation::chat::{request::ChatMessageRequest, ChatMessage, MessageRole},
    models::ModelOptions as GenerationOptions,
    Ollama as OllamaClient,
};
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::StreamExt;

#[derive(Debug, Clone)]
pub struct Ollama {
    pub(crate) client: Arc<OllamaClient>,
    pub(crate) model: String,
    pub(crate) options: Option<GenerationOptions>,
}

/// [llama3.2](https://ollama.com/library/llama3.2) is a 3B parameters, 2.0GB model.
const DEFAULT_MODEL: &str = "llama3.2";

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
        let request = ChatMessageRequest::new(self.model.clone(), mapped_messages);
        match &self.options {
            Some(options) => request.options(options.clone()),
            None => request,
        }
    }
}

impl From<&Message> for ChatMessage {
    fn from(message: &Message) -> Self {
        let images = match message.images.clone() {
            Some(images) => {
                let images = images
                    .iter()
                    .map(|image| Image::from_base64(&image.image_url))
                    .collect();
                Some(images)
            }
            None => None,
        };
        let mut chat_message =
            ChatMessage::new(message.message_type.clone().into(), message.content.clone());
        chat_message.images = images;
        chat_message
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

fn token_usage(final_data: &ChatMessageFinalResponseData) -> TokenUsage {
    let prompt_tokens = final_data.prompt_eval_count as u32;
    let completion_tokens = final_data.eval_count as u32;
    TokenUsage::new(prompt_tokens, completion_tokens)
}

#[async_trait]
impl LLM for Ollama {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let request = self.generate_request(messages);
        let result = self.client.send_chat_messages(request).await?;

        let generation = result.message.content;

        let tokens = result.final_data.as_ref().map(token_usage);

        Ok(GenerateResult { tokens, generation })
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let request = self.generate_request(messages);
        let result = self.client.send_chat_messages_stream(request).await?;

        let stream = result.map(|data| match data {
            Ok(data) => {
                let content = data.message.content.clone();
                let tokens = data.final_data.as_ref().map(token_usage);
                Ok(StreamData::new(
                    serde_json::to_value(data).unwrap_or_default(),
                    tokens,
                    content,
                ))
            }
            Err(_) => Err(OllamaError::Other("Stream error".to_string()).into()),
        });

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
        let ollama = Ollama::default().with_model("llama3.2");
        let response = ollama.invoke("Hey Macarena, ay").await.unwrap();
        println!("{}", response);
    }

    #[tokio::test]
    #[ignore]
    async fn test_stream() {
        let ollama = Ollama::default().with_model("llama3.2");

        let message = Message::new_human_message("Why does water boil at 100 degrees?");
        let mut stream = ollama.stream(&vec![message]).await.unwrap();
        let mut stdout = tokio::io::stdout();
        while let Some(res) = stream.next().await {
            let data = res.unwrap();
            stdout.write_all(data.content.as_bytes()).await.unwrap();
        }
        stdout.write_all(b"\n").await.unwrap();
        stdout.flush().await.unwrap();
    }

    fn ollama_for(server: &mockito::ServerGuard) -> Ollama {
        let client = Arc::new(OllamaClient::try_new(server.url()).unwrap());
        Ollama::new(client, "m", None)
    }

    #[tokio::test]
    async fn generate_against_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "model": "m",
                "stream": false,
                "options": {"temperature": 0.5},
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"}
                ]
            })))
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "model": "m",
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": "hello"},
                    "done": true,
                    "total_duration": 1,
                    "load_duration": 1,
                    "prompt_eval_count": 3,
                    "prompt_eval_duration": 1,
                    "eval_count": 2,
                    "eval_duration": 1
                })
                .to_string(),
            )
            .create_async()
            .await;

        let result = ollama_for(&server)
            .with_options(GenerationOptions::default().temperature(0.5))
            .generate(&[
                Message::new_system_message("sys"),
                Message::new_human_message("hi"),
            ])
            .await
            .unwrap();
        mock.assert_async().await;
        assert_eq!(result.generation, "hello");
        let tokens = result.tokens.unwrap();
        assert_eq!(
            (
                tokens.prompt_tokens,
                tokens.completion_tokens,
                tokens.total_tokens
            ),
            (3, 2, 5)
        );
    }

    #[tokio::test]
    async fn stream_against_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let line = |content: &str, done: bool| {
            let mut v = serde_json::json!({
                "model": "m",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": content},
                "done": done
            });
            if done {
                for (k, n) in [
                    ("total_duration", 1),
                    ("load_duration", 1),
                    ("prompt_eval_count", 3),
                    ("prompt_eval_duration", 1),
                    ("eval_count", 2),
                    ("eval_duration", 1),
                ] {
                    v[k] = n.into();
                }
            }
            v.to_string()
        };
        server
            .mock("POST", "/api/chat")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"stream": true}),
            ))
            .with_header("content-type", "application/x-ndjson")
            .with_body(format!("{}\n{}\n", line("Hel", false), line("lo", true)))
            .create_async()
            .await;

        let mut stream = ollama_for(&server)
            .stream(&[Message::new_human_message("hi")])
            .await
            .unwrap();
        let mut text = String::new();
        let mut usage = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            text.push_str(&chunk.content);
            usage = chunk.tokens.or(usage);
        }
        assert_eq!(text, "Hello");
        assert_eq!(usage.unwrap().total_tokens, 5);
    }
}
