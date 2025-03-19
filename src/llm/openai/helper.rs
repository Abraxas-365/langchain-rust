use async_openai::{
    error::OpenAIError,
    types::{
        ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessageArgs,
        ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImageArgs,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestToolMessageArgs,
        ChatCompletionRequestUserMessageArgs, ChatCompletionRequestUserMessageContent,
    },
};

use crate::{
    language_models::LLMError,
    schemas::{Message, MessageType},
};

fn to_openai_message(message: &Message) -> Result<ChatCompletionRequestMessage, LLMError> {
    match message.message_type {
        MessageType::AIMessage => Ok(match &message.tool_calls {
            Some(value) => {
                let function: Vec<ChatCompletionMessageToolCall> = value.clone();
                ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(function)
                    .content(message.content.clone())
                    .build()?
                    .into()
            }
            None => ChatCompletionRequestAssistantMessageArgs::default()
                .content(message.content.clone())
                .build()?
                .into(),
        }),
        MessageType::HumanMessage => {
            let content: ChatCompletionRequestUserMessageContent = match message.images.clone() {
                Some(images) => images
                    .into_iter()
                    .map(|image| {
                        ChatCompletionRequestMessageContentPartImageArgs::default()
                            .image_url(image.image_url)
                            .build()
                            .map(Into::into)
                    })
                    .collect::<Result<Vec<_>, OpenAIError>>()?
                    .into(),
                None => message.content.clone().into(),
            };

            Ok(ChatCompletionRequestUserMessageArgs::default()
                .content(content)
                .build()?
                .into())
        }
        MessageType::SystemMessage => Ok(ChatCompletionRequestSystemMessageArgs::default()
            .content(message.content.clone())
            .build()?
            .into()),
        MessageType::ToolMessage => Ok(ChatCompletionRequestToolMessageArgs::default()
            .content(message.content.clone())
            .tool_call_id(message.id.clone().unwrap_or_default())
            .build()?
            .into()),
    }
}

pub fn to_openai_messages(
    messages: &[Message],
) -> Result<Vec<ChatCompletionRequestMessage>, LLMError> {
    messages.iter().map(to_openai_message).collect()
}
