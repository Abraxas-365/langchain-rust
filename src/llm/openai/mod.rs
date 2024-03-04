use std::{error::Error, pin::Pin, sync::Arc};

use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionFunctions, ChatCompletionFunctionsArgs,
        ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequest, CreateChatCompletionRequestArgs,
    },
    Client,
};
use async_trait::async_trait;
use futures::{Future, StreamExt};
use tokio::sync::Mutex;

use crate::{
    language_models::{
        llm::LLM,
        options::{CallOptions, FunctionCallBehavior, FunctionDefinition},
        GenerateResult, TokenUsage,
    },
    schemas::messages::{Message, MessageType},
};

#[derive(Clone)]
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

#[derive(Clone)]
pub struct OpenAI {
    config: OpenAIConfig,
    model: String,
    stop_words: Option<Vec<String>>,
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

impl Into<Box<dyn LLM>> for OpenAI {
    fn into(self) -> Box<dyn LLM> {
        Box::new(self)
    }
}

impl Default for OpenAI {
    fn default() -> Self {
        Self {
            config: OpenAIConfig::default(),
            model: OpenAIModel::Gpt35.to_string(),
            stop_words: None,
            max_tokens: 256,
            temperature: 0.0,
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
            model: OpenAIModel::Gpt35.to_string(),
            stop_words: opt.stop_words,
            max_tokens: opt.max_tokens.unwrap_or(256),
            temperature: opt.temperature.unwrap_or(0.0),
            top_p: opt.top_p.unwrap_or(1.0),
            frequency_penalty: opt.frequency_penalty.unwrap_or(0.0),
            presence_penalty: opt.presence_penalty.unwrap_or(0.0),
            function_call_behavior: opt.function_call_behavior,
            functions: opt.functions,
            streaming_func: opt.streaming_func,
        }
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.config = self.config.with_api_key(api_key);
        self
    }

    pub fn with_api_base<S: Into<String>>(mut self, api_base: S) -> Self {
        self.config = self.config.with_api_base(api_base);
        self
    }

    pub fn with_org_id<S: Into<String>>(mut self, org_id: S) -> Self {
        self.config = self.config.with_org_id(org_id);
        self
    }
}

#[async_trait]
impl LLM for OpenAI {
    async fn generate(&self, prompt: &[Message]) -> Result<GenerateResult, Box<dyn Error>> {
        let client = Client::with_config(self.config.clone());
        let request = self.generate_request(prompt)?;
        match &self.streaming_func {
            Some(func) => {
                let mut stream = client.chat().create_stream(request).await?;
                let mut complete_response = String::new();
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(response) => {
                            for chat_choice in response.choices.iter() {
                                if let Some(ref content) = chat_choice.delta.content {
                                    let mut func = func.lock().await;
                                    let _ = func(content.clone()).await;
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

    async fn invoke(&self, prompt: &str) -> Result<String, Box<dyn Error>> {
        self.generate(&[Message::new_human_message(prompt)])
            .await
            .map(|res| res.generation)
    }

    fn with_options(&mut self, options: CallOptions) {
        self.max_tokens = options.max_tokens.unwrap_or(256);
        self.stop_words = options.stop_words;
        self.temperature = options.temperature.unwrap_or(0.7);
        self.top_p = options.top_p.unwrap_or(1.0);
        self.frequency_penalty = options.frequency_penalty.unwrap_or(0.0);
        self.presence_penalty = options.presence_penalty.unwrap_or(0.0);
        self.function_call_behavior = options.function_call_behavior;
        self.functions = options.functions;
        self.streaming_func = options.streaming_func;
    }
}

impl OpenAI {
    fn to_openai_messages(
        &self,
        messages: &[Message],
    ) -> Result<Vec<ChatCompletionRequestMessage>, Box<dyn Error>> {
        let mut openai_messages: Vec<ChatCompletionRequestMessage> = Vec::new();
        for m in messages {
            match m.message_type {
                MessageType::AIMessage => openai_messages.push(
                    ChatCompletionRequestAssistantMessageArgs::default()
                        .content(m.content.clone())
                        .build()?
                        .into(),
                ),
                MessageType::HumanMessage => openai_messages.push(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(m.content.clone())
                        .build()?
                        .into(),
                ),
                MessageType::SystemMessage => openai_messages.push(
                    ChatCompletionRequestSystemMessageArgs::default()
                        .content(m.content.clone())
                        .build()?
                        .into(),
                ),
            }
        }
        Ok(openai_messages)
    }

    fn generate_request(
        &self,
        messages: &[Message],
    ) -> Result<CreateChatCompletionRequest, Box<dyn Error>> {
        let messages: Vec<ChatCompletionRequestMessage> = self.to_openai_messages(messages)?;
        let mut request_builder = CreateChatCompletionRequestArgs::default();
        request_builder.max_tokens(self.max_tokens);
        request_builder.model(self.model.to_string());
        if let Some(stop_words) = &self.stop_words {
            request_builder.stop(stop_words);
        }

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
        Ok(request_builder.build()?)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_ivoke() {
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
        let options = CallOptions::new().with_streaming_func(streaming_func);
        // Setup the OpenAI client with the necessary options
        let open_ai = OpenAI::new(options).with_model(OpenAIModel::Gpt35.to_string()); // You can change the model as needed

        // Define a set of messages to send to the generate function

        // Call the generate function
        match open_ai.invoke("hola").await {
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
        let open_ai = OpenAI::new(options).with_model(OpenAIModel::Gpt35.to_string()); // You can change the model as needed

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
