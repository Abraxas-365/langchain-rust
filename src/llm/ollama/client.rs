use crate::language_models::options::CallOptions;
use ollama_rs::generation::functions::tools::Tool as OllamaTool;
use crate::tools::Tool;
use crate::{
    language_models::{llm::LLM, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, MessageType, StreamData},
};
use async_trait::async_trait;
use futures::Stream;
use ollama_rs::generation::functions::{FunctionCallRequest, LlamaFunctionCall};
use ollama_rs::generation::images::Image;
pub use ollama_rs::{
    error::OllamaError,
    generation::{
        chat::{request::ChatMessageRequest, ChatMessage, MessageRole},
        options::GenerationOptions,
    },
    Ollama as OllamaClient,
};
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::StreamExt;

enum OllamaRequest {
    ChatMessageRequest(ChatMessageRequest),
    FunctionCallRequest(FunctionCallRequest),
}

pub struct OllamaToolStruct{
    pub(crate) tool: Arc<dyn Tool>,
}

#[async_trait]
impl OllamaTool for OllamaToolStruct {
    fn name(&self) -> String {
        self.tool.name()
    }

    fn description(&self) -> String {
        self.tool.description()
    }

    fn parameters(&self) -> serde_json::Value {
        self.tool.parameters()
    }

    async fn run(&self, input: serde_json::Value) -> Result<String, Box<dyn std::error::Error>> {
        self.tool.run(input).await
    }
}

#[derive(Clone)]
pub struct Ollama {
    pub(crate) client: Arc<OllamaClient>,
    pub(crate) model: String,
    pub(crate) options: CallOptions,
}

/// [llama3.2](https://ollama.com/library/llama3.2) is a 3B parameters, 2.0GB model.
const DEFAULT_MODEL: &str = "llama3.2";

impl Ollama {
    pub fn new<S: Into<String>>(client: Arc<OllamaClient>, model: S, options: CallOptions) -> Self {
        Ollama {
            client,
            model: model.into(),
            options,
        }
    }

    fn with_options(mut self, options: CallOptions) -> Self {
        self.options = options;
        self
    }

    fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    fn generate_options(&self) -> GenerationOptions {
        let mut options = GenerationOptions::default();
        if let Some(mirostat) = self.options.mirostat {
            options = options.mirostat(mirostat);
        }
        if let Some(mirostat_eta) = self.options.mirostat_eta {
            options = options.mirostat_eta(mirostat_eta);
        }
        if let Some(mirostat_tau) = self.options.mirostat_tau {
            options = options.mirostat_tau(mirostat_tau);
        }
        if let Some(num_ctx) = self.options.num_ctx {
            options = options.num_ctx(num_ctx);
        }
        if let Some(num_gqa) = self.options.num_gqa {
            options = options.num_gqa(num_gqa);
        }
        if let Some(num_gpu) = self.options.num_gpu {
            options = options.num_gpu(num_gpu);
        }
        if let Some(num_thread) = self.options.num_thread {
            options = options.num_thread(num_thread);
        }
        if let Some(repeat_last_n) = self.options.repeat_last_n {
            options = options.repeat_last_n(repeat_last_n);
        }
        if let Some(repeat_penalty) = self.options.repetition_penalty {
            options = options.repeat_penalty(repeat_penalty);
        }
        if let Some(temperature) = self.options.temperature {
            options = options.temperature(temperature);
        }
        if let Some(seed) = self.options.seed {
            options = options.seed(seed as i32);
        }
        if let Some(stop) = &self.options.stop_words {
            options = options.stop(stop.clone());
        }
        if let Some(tfs_z) = self.options.tfs_z {
            options = options.tfs_z(tfs_z);
        }
        if let Some(num_predict) = self.options.num_predict {
            options = options.num_predict(num_predict);
        }
        if let Some(top_k) = self.options.top_k {
            options = options.top_k(top_k as u32);
        }
        if let Some(top_p) = self.options.top_p {
            options = options.top_p(top_p);
        }
        options
    }
    #[cfg(feature = "ollama")]
    fn generate_request(&self, messages: &[Message]) -> OllamaRequest {
        let options = self.generate_options();
        let mapped_messages = messages.iter().map(|message| message.into()).collect();
        if let Some(tools) = self.options.functions.clone() {
            let tools = tools
                .into_iter()
                .map(|tool| Arc::new(OllamaToolStruct { tool: tool.clone() }) as Arc<dyn OllamaTool>)
                .collect();
            OllamaRequest::FunctionCallRequest(
                FunctionCallRequest::new(self.model.clone(), tools, mapped_messages)
                    .options(options),
            )
        } else {
            OllamaRequest::ChatMessageRequest(
                ChatMessageRequest::new(self.model.clone(), mapped_messages).options(options),
            )
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
        ChatMessage {
            content: message.content.clone(),
            images,
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
        Ollama::new(client, String::from(DEFAULT_MODEL), CallOptions::default())
    }
}

#[async_trait]
impl LLM for Ollama {
    fn add_options(&mut self, options: CallOptions) {
        self.options.merge_options(options);
    }

    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let request = self.generate_request(messages);
        let result = match request {
            OllamaRequest::ChatMessageRequest(request) => {
                self.client.send_chat_messages(request).await?
            }
            OllamaRequest::FunctionCallRequest(request) => {
                self.client
                    .send_function_call(request, Arc::new(LlamaFunctionCall {}))
                    .await?
            }
        };
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
        let result = match request {
            OllamaRequest::ChatMessageRequest(request) => {
                self.client.send_chat_messages_stream(request).await?
            }
            OllamaRequest::FunctionCallRequest(_) => {
                return Err(LLMError::OtherError(
                    "Function call stream not supported".to_string(),
                ));
            }
        };
        let stream = result.map(|data| match data {
            Ok(data) => match data.message.clone() {
                Some(message) => Ok(StreamData::new(
                    serde_json::to_value(data).unwrap_or_default(),
                    message.content,
                )),
                // TODO: no need to return error, see https://github.com/Abraxas-365/langchain-rust/issues/140
                None => Err(LLMError::ContentNotFound(
                    "No message in response".to_string(),
                )),
            },
            Err(_) => Err(OllamaError::from("Stream error".to_string()).into()),
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
            stdout.write(data.content.as_bytes()).await.unwrap();
        }
        stdout.write(b"\n").await.unwrap();
        stdout.flush().await.unwrap();
    }
}
