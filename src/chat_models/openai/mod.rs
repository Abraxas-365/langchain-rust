use std::{error::Error, pin::Pin, sync::Arc};

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionFunctions, ChatCompletionFunctionsArgs,
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use async_trait::async_trait;
use futures::{Future, StreamExt};
use tokio::sync::Mutex;

use crate::{
    language_models::{
        chat_model::LLMChat,
        options::{CallOptions, FunctionCallBehavior, FunctionDefinition},
        GenerateResult, TokenUsage,
    },
    schemas::messages::{Message, MessageType},
};

pub enum OpenAIModel {
    Gpt35,
    Gpt4,
    Gpt4Turbo,
}

impl ToString for OpenAIModel {
    fn to_string(&self) -> String {
        match self {
            OpenAIModel::Gpt35 => "gpt-3.5-turbo".to_string(),
            OpenAIModel::Gpt4 => "gpt-4".to_string(),
            OpenAIModel::Gpt4Turbo => "gpt-4-turbo-preview".to_string(),
        }
    }
}

pub struct OpenAI {
    config: OpenAIConfig,
    model: OpenAIModel,
    max_tokens: u16,
    temperature: f32,
    top_p: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    function_call_behavior: Option<FunctionCallBehavior>,
    functions: Option<Vec<FunctionDefinition>>,
    streaming_func: Option<
        Arc<
            Mutex<dyn FnMut(String) -> Pin<Box<dyn Future<Output = Result<(), ()>> + Send>> + Send>,
        >,
    >,
}

impl Default for OpenAI {
    fn default() -> Self {
        Self {
            config: OpenAIConfig::default(),
            model: OpenAIModel::Gpt35,
            max_tokens: 256,
            temperature: 0.7,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            function_call_behavior: None,
            functions: None,
            streaming_func: None,
        }
    }
}

impl OpenAI {
    pub fn new(opt: CallOptions) -> Self {
        Self {
            config: OpenAIConfig::default(),
            model: OpenAIModel::Gpt35,
            max_tokens: opt.max_tokens.unwrap_or(256),
            temperature: opt.temperature.unwrap_or(0.7),
            top_p: opt.top_p.unwrap_or(1.0),
            frequency_penalty: opt.frequency_penalty.unwrap_or(0.0),
            presence_penalty: opt.presence_penalty.unwrap_or(0.0),
            function_call_behavior: opt.function_call_behavior,
            functions: opt.functions,
            streaming_func: opt.streaming_func,
        }
    }

    pub fn with_model(mut self, model: OpenAIModel) -> Self {
        self.model = model;
        self
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.config = OpenAIConfig::new().with_api_key(api_key);
        self
    }
}

#[async_trait]
impl LLMChat for OpenAI {
    async fn generate(&self, prompt: &[Message]) -> Result<GenerateResult, Box<dyn Error>> {
        let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();
        for m in prompt {
            match m.message_type {
                MessageType::AIMessage => messages.push(
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(m.content.clone())
                        .build()?
                        .into(),
                ),
                MessageType::HumanMessage => messages.push(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(m.content.clone())
                        .build()?
                        .into(),
                ),
                MessageType::SystemMessage => messages.push(
                    ChatCompletionRequestSystemMessageArgs::default()
                        .content(m.content.clone())
                        .build()?
                        .into(),
                ),
            }
        }
        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder.max_tokens(self.max_tokens);
        request_builder.model(self.model.to_string());

        if let Some(behavior) = &self.functions {
            let mut functions: Vec<ChatCompletionFunctions> = Vec::new();
            for f in behavior.iter() {
                let tool = ChatCompletionFunctionsArgs::default()
                    .name(f.name.clone())
                    .description(f.description.clone())
                    .parameters(f.parameters.clone())
                    .build()?;
                functions.push(tool);
            }
            request_builder.functions(functions);
        }

        if let Some(behavior) = self.function_call_behavior {
            match behavior {
                FunctionCallBehavior::Auto => request_builder.function_call("auto"),
                FunctionCallBehavior::None => request_builder.function_call("none"),
            };
        }
        request_builder.messages(messages);
        let request = request_builder
            .build()
            .map_err(|e| Box::new(e) as Box<dyn Error>)?;

        let client = Client::with_config(self.config.clone());
        match &self.streaming_func {
            Some(func) => {
                let mut stream = client.chat().create_stream(request).await?;
                let mut complete_response = String::new();
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(response) => {
                            for chat_choice in response.choices.iter() {
                                if let Some(ref content) = chat_choice.delta.content {
                                    let mut func = func.lock().await; // Assuming this is a Tokio Mutex
                                    if let Err(e) = func(content.clone()).await {
                                        eprintln!("Error calling streaming function: {:?}", e);
                                    }
                                    complete_response.push_str(content);
                                }
                            }
                        }
                        Err(err) => {
                            eprintln!("Error from streaming response: {:?}", err);
                        }
                    }
                }
                let mut generate_result = GenerateResult::default();
                generate_result.generation = complete_response;
                Ok(generate_result)
            }
            None => {
                let response = client.chat().create(request).await?;
                let mut generate_result = GenerateResult::default();

                if let Some(usage) = response.usage {
                    generate_result.tokens = Some(TokenUsage {
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                        total_tokens: usage.total_tokens,
                    });
                }

                if let Some(choice) = response.choices.first() {
                    generate_result.generation = choice.message.content.clone().unwrap_or_default();
                } else {
                    generate_result.generation = "".to_string();
                }

                Ok(generate_result)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_generate_function() {
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
        // Define the streaming function as an async block without capturing external references directly
        let options = CallOptions::new().with_streaming_func(streaming_func);
        // Setup the OpenAI client with the necessary options
        let open_ai = OpenAI::new(options).with_model(OpenAIModel::Gpt35); // You can change the model as needed

        // Define a set of messages to send to the generate function
        let messages = vec![Message::new_human_message("Hello, how are you?")];

        // Call the generate function
        match open_ai.generate(&messages).await {
            Ok(result) => {
                // Print the response from the generate function
                println!("Generate Result: {:?}", result);
                println!("Message Complete: {:?}", message_complete.lock().await);
            }
            Err(e) => {
                // Handle any errors
                eprintln!("Error calling generate: {:?}", e);
            }
        }
    }
}
