use std::{error::Error, pin::Pin};

pub use async_openai::config::{AzureConfig, Config, OpenAIConfig};
use async_openai::{
    types::{
        ChatChoiceStream, ChatCompletionFunctionsArgs, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest,
        CreateChatCompletionRequestArgs,
    },
    Client,
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};

use crate::{
    language_models::{
        llm::LLM,
        options::{CallOptions, FunctionCallBehavior},
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

impl Into<String> for OpenAIModel {
    fn into(self) -> String {
        self.to_string()
    }
}

#[derive(Clone)]
pub struct OpenAI<C: Config> {
    config: C,
    options: CallOptions,
    model: String,
}

impl<C: Config + Send + Sync + 'static> Into<Box<dyn LLM>> for OpenAI<C> {
    fn into(self) -> Box<dyn LLM> {
        Box::new(self)
    }
}

impl<C: Config> OpenAI<C> {
    pub fn new(config: C) -> Self {
        Self {
            config,
            options: CallOptions::default(),
            model: OpenAIModel::Gpt35.to_string(),
        }
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_config(mut self, config: C) -> Self {
        self.config = config;
        self
    }

    pub fn with_options(mut self, options: CallOptions) -> Self {
        self.options = options;
        self
    }
}

impl Default for OpenAI<OpenAIConfig> {
    fn default() -> Self {
        Self::new(OpenAIConfig::default())
    }
}

#[async_trait]
impl<C: Config + Send + Sync> LLM for OpenAI<C> {
    async fn generate(&self, prompt: &[Message]) -> Result<GenerateResult, Box<dyn Error>> {
        let client = Client::with_config(self.config.clone());
        let request = self.generate_request(prompt)?;
        match &self.options.streaming_func {
            Some(func) => {
                let mut stream = client.chat().create_stream(request).await?;
                let mut complete_response = String::new();
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(response) => {
                            for chat_choice in response.choices.iter() {
                                let chat_choice: ChatChoiceStream = chat_choice.clone();
                                {
                                    let mut func = func.lock().await;
                                    let _ = func(
                                        serde_json::to_string(&chat_choice).unwrap_or("".into()),
                                    )
                                    .await;
                                }
                                if let Some(content) = chat_choice.delta.content {
                                    complete_response.push_str(&content);
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

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<serde_json::Value, Box<dyn Error + Send>>> + Send>>,
        Box<dyn Error>,
    > {
        let client = Client::with_config(self.config.clone());
        let request = self.generate_request(messages)?;

        let original_stream = client.chat().create_stream(request).await?;

        let new_stream = original_stream.map(|result| match result {
            Ok(completion) => {
                serde_json::to_value(completion).map_err(|e| Box::new(e) as Box<dyn Error + Send>)
            }
            Err(e) => Err(Box::new(e) as Box<dyn Error + Send>),
        });

        Ok(Box::pin(new_stream))
    }

    fn add_options(&mut self, options: CallOptions) {
        self.options = options;
    }
}

impl<C: Config> OpenAI<C> {
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
        if let Some(max_tokens) = self.options.max_tokens {
            request_builder.max_tokens(max_tokens);
        }
        request_builder.model(self.model.to_string());
        if let Some(stop_words) = &self.options.stop_words {
            request_builder.stop(stop_words);
        }

        if let Some(behavior) = &self.options.functions {
            let mut functions = Vec::new();
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

        if let Some(behavior) = self.options.function_call_behavior {
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
    use std::sync::Arc;
    use tokio::sync::Mutex;
    use tokio::test;

    #[test]
    #[ignore]
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
        let open_ai = OpenAI::new(OpenAIConfig::default())
            .with_model(OpenAIModel::Gpt35.to_string()) // You can change the model as needed
            .with_options(options);

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
    #[ignore]
    async fn test_generate_function() {
        let message_complete = Arc::new(Mutex::new(String::new()));

        // Define the streaming function
        // This function will append the content received from the stream to `message_complete`
        let streaming_func = {
            let message_complete = message_complete.clone();
            move |content: String| {
                let message_complete = message_complete.clone();
                async move {
                    let content = serde_json::from_str::<ChatChoiceStream>(&content).unwrap();
                    if content.finish_reason.is_some() {
                        return Ok(());
                    }
                    let mut message_complete_lock = message_complete.lock().await;
                    println!("Content: {:?}", content);
                    message_complete_lock.push_str(&content.delta.content.unwrap());
                    Ok(())
                }
            }
        };
        // Define the streaming function as an async block without capturing external references directly
        let options = CallOptions::new().with_streaming_func(streaming_func);
        // Setup the OpenAI client with the necessary options
        let open_ai = OpenAI::new(OpenAIConfig::default())
            .with_model(OpenAIModel::Gpt35.to_string()) // You can change the model as needed
            .with_options(options);

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

    #[test]
    #[ignore]
    async fn test_openai_stream() {
        // Setup the OpenAI client with the necessary options
        let open_ai = OpenAI::default().with_model(OpenAIModel::Gpt35.to_string());

        // Define a set of messages to send to the generate function
        let messages = vec![Message::new_human_message("Hello, how are you?")];

        open_ai
            .stream(&messages)
            .await
            .unwrap()
            .for_each(|result| async {
                match result {
                    Ok(response) => {
                        if let Some(content) = response.pointer("/choices/0/delta/content") {
                            println!("Content: {}", content.as_str().unwrap_or(""));
                        }
                    }
                    Err(e) => {
                        eprintln!("Error: {:?}", e);
                    }
                }
            })
            .await;
    }
}
