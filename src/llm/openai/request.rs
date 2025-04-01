use async_openai::{
    error::OpenAIError,
    types::{
        ChatCompletionRequestMessage, ChatCompletionStreamOptions, ChatCompletionTool,
        ChatCompletionToolChoiceOption, ResponseFormat,
    },
};
use serde::Serialize;

use crate::{
    language_models::{options::CallOptions, LLMError},
    schemas::Message,
};

use super::helper::to_openai_messages;

#[derive(Serialize, Debug)]
pub struct OpenAIRequest {
    pub messages: Vec<ChatCompletionRequestMessage>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<ChatCompletionStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_length: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatCompletionTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatCompletionToolChoiceOption>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

impl OpenAIRequest {
    pub fn build_request<S: Into<String>>(
        model: S,
        messages: Vec<Message>,
        call_options: &CallOptions,
    ) -> Result<OpenAIRequest, LLMError> {
        let messages = to_openai_messages(messages)?;

        Ok(OpenAIRequest {
            messages,
            model: model.into(),
            stream: Some(call_options.stream_option.is_some()),
            stream_options: call_options.stream_option.as_ref().map(|stream| {
                ChatCompletionStreamOptions {
                    include_usage: stream.include_usage,
                }
            }),
            candidate_count: call_options.candidate_count,
            max_tokens: call_options.max_tokens,
            temperature: call_options.temperature,
            stop: call_options.stop_words.clone(),
            top_k: call_options.top_k,
            top_p: call_options.top_p,
            seed: call_options.seed,
            min_length: call_options.min_length,
            max_length: call_options.max_length,
            n: call_options.n,
            repetition_penalty: call_options.repetition_penalty,
            frequency_penalty: call_options.frequency_penalty,
            presence_penalty: call_options.presence_penalty,
            tools: call_options
                .functions
                .clone()
                .map(|fs| {
                    fs.into_iter()
                        .map(|f| f.try_into())
                        .collect::<Result<Vec<_>, OpenAIError>>()
                })
                .transpose()?,
            tool_choice: call_options.tool_choice.clone(),
            response_format: call_options.response_format.clone().map(|r| r.into()),
        })
    }
}
