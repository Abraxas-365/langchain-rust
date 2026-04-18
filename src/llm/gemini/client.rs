use async_trait::async_trait;
use futures::{Stream, StreamExt};
use reqwest::Client;

use std::pin::Pin;

use crate::{
    language_models::{llm::LLM, options::CallOptions, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, MessageType, StreamData},
};

use super::models::{
    GeminiContent, GeminiGenerateRequest, GeminiPart, GeminiResponse, GenerationConfig,
};

/// Common Google Gemini LLM Models.
#[derive(Clone)]
pub enum GeminiModel {
    /// Gemini 2.0 Flash (Default)
    Gemini2_0Flash,
    /// Gemini 1.5 Flash
    Gemini1_5Flash,
    /// Gemini 1.5 Pro
    Gemini1_5Pro,
}

impl ToString for GeminiModel {
    fn to_string(&self) -> String {
        match self {
            GeminiModel::Gemini2_0Flash => "gemini-2.0-flash".to_string(),
            GeminiModel::Gemini1_5Flash => "gemini-1.5-flash".to_string(),
            GeminiModel::Gemini1_5Pro => "gemini-1.5-pro".to_string(),
        }
    }
}

impl From<GeminiModel> for String {
    fn from(val: GeminiModel) -> Self {
        val.to_string()
    }
}

/// Google Gemini chat model client.
///
/// Implements the [`LLM`] trait for use in chains and agents.
/// Authentication is provided via API key — optionally set via `GEMINI_API_KEY` environment variable.
///
/// # Example
/// ```rust,no_run
/// use langchain_rust::llm::gemini::client::{Gemini, GeminiModel};
///
/// let llm = Gemini::builder()
///     .model(GeminiModel::Gemini2_0Flash)
///     .build()
///     .unwrap();
/// ```
#[derive(Clone)]
pub struct Gemini {
    api_key: String,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    client: Client,
    options: CallOptions,
}

/// A builder to configure and instantiate a [`Gemini`] client.
#[derive(Default)]
pub struct GeminiBuilder {
    api_key: Option<String>,
    model: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    client: Option<Client>,
    options: Option<CallOptions>,
}

impl GeminiBuilder {
    /// Creates a new, empty builder pattern instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the API key explicitly. Use `GEMINI_API_KEY` for implicit detection.
    pub fn api_key<S: Into<String>>(mut self, api_key: S) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the model name representation. Can be a string or `GeminiModel`.
    pub fn model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the sampling temperature (e.g., 0.7) for deterministic token randomness.
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the absolute token count limit for the response.
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Sets Top-P boundary configuration logic.
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Sets Top-K boundary configuration logic.
    pub fn top_k(mut self, top_k: u32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Provide a pre-existing custom `reqwest::Client`.
    pub fn client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Complete the construct and finalize the `Gemini` client.
    ///
    /// # Errors
    /// Returns an error if the API key is absent both here and in `GEMINI_API_KEY`.
    pub fn build(self) -> Result<Gemini, String> {
        let api_key = self
            .api_key
            .or_else(|| std::env::var("GEMINI_API_KEY").ok())
            .ok_or_else(|| {
                "No API key found. Provide via builder or GEMINI_API_KEY env var".to_string()
            })?;

        Ok(Gemini {
            api_key,
            model: self
                .model
                .unwrap_or_else(|| GeminiModel::Gemini2_0Flash.to_string()),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            top_p: self.top_p,
            top_k: self.top_k,
            client: self.client.unwrap_or_default(),
            options: self.options.unwrap_or_default(),
        })
    }
}

impl Gemini {
    /// Bootstraps to configure the provider client seamlessly.
    pub fn builder() -> GeminiBuilder {
        GeminiBuilder::new()
    }

    /// Direct fast-instantiation constructor taking only the key and model.
    pub fn new<S: Into<String>>(api_key: S, model: S) -> Result<Self, String> {
        Self::builder().api_key(api_key).model(model).build()
    }

    /// Mutates the model setting inline.
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    /// Validates and marshalls prompt message inputs into a proper Google Gemini formatted payload.
    fn map_messages(&self, messages: &[Message]) -> (Vec<GeminiContent>, Option<GeminiContent>) {
        let mut contents = Vec::new();
        let mut system_instruction = None;

        for m in messages {
            match m.message_type {
                MessageType::SystemMessage => {
                    system_instruction = Some(GeminiContent {
                        role: "system".to_string(), // Safely abstracted by Gemini implementation top level format
                        parts: vec![GeminiPart {
                            text: m.content.clone(),
                        }],
                    });
                }
                MessageType::HumanMessage => {
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart {
                            text: m.content.clone(),
                        }],
                    });
                }
                MessageType::AIMessage => {
                    contents.push(GeminiContent {
                        role: "model".to_string(),
                        parts: vec![GeminiPart {
                            text: m.content.clone(),
                        }],
                    });
                }
                _ => {
                    // Default fallback treating context blocks as user messages typically
                    contents.push(GeminiContent {
                        role: "user".to_string(),
                        parts: vec![GeminiPart {
                            text: m.content.clone(),
                        }],
                    });
                }
            }
        }

        (contents, system_instruction)
    }

    /// Encapsulates generation parameter overrides for dynamic runtime context.
    fn build_config(&self) -> Option<GenerationConfig> {
        let temp = self.options.temperature.or(self.temperature);
        let max_tok = self.options.max_tokens.or(self.max_tokens);
        let top_p = self.options.top_p.or(self.top_p);
        let top_k = self.options.top_k.map(|k| k as u32).or(self.top_k);

        if temp.is_some() || max_tok.is_some() || top_p.is_some() || top_k.is_some() {
            Some(GenerationConfig {
                temperature: temp,
                max_output_tokens: max_tok,
                top_p,
                top_k,
            })
        } else {
            None
        }
    }

    /// Centralized payload builder conforming to API request standard.
    fn build_request(&self, messages: &[Message]) -> GeminiGenerateRequest {
        let (contents, system_instruction) = self.map_messages(messages);
        let generation_config = self.build_config();

        GeminiGenerateRequest {
            contents,
            system_instruction,
            generation_config,
            safety_settings: None,
        }
    }
}

impl Default for Gemini {
    fn default() -> Self {
        Self::builder().build().unwrap_or_else(|_| Gemini {
            api_key: String::new(),
            model: GeminiModel::Gemini2_0Flash.to_string(),
            temperature: None,
            max_tokens: None,
            top_p: None,
            top_k: None,
            client: Client::default(),
            options: CallOptions::default(),
        })
    }
}

#[async_trait]
impl LLM for Gemini {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        if self.options.streaming_func.is_some() {
            let mut complete_response = String::new();
            let mut stream = self.stream(messages).await?;
            while let Some(data) = stream.next().await {
                match data {
                    Ok(value) => {
                        if let Some(func) = self.options.streaming_func.clone() {
                            let mut func = func.lock().await;
                            complete_response.push_str(&value.content);
                            let _ = func(value.content).await;
                        } else {
                            complete_response.push_str(&value.content);
                        }
                    }
                    Err(e) => return Err(e),
                }
            }
            return Ok(GenerateResult {
                generation: complete_response,
                ..Default::default()
            });
        }

        let req_body = self.build_request(messages);
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
            self.model, self.api_key
        );

        let res = self
            .client
            .post(&url)
            .json(&req_body)
            .send()
            .await
            .map_err(|e| {
                LLMError::OtherError(format!("Failed to execute Gemini HTTP Request: {:?}", e))
            })?;

        if !res.status().is_success() {
            let status = res.status();
            let text = res
                .text()
                .await
                .unwrap_or_else(|_| "Unavailable response body text".to_string());
            return Err(LLMError::OtherError(format!(
                "Gemini API Error: {} - {}",
                status, text
            )));
        }

        let resp: GeminiResponse = res.json().await.map_err(|e| {
            LLMError::OtherError(format!("Gemini API JSON response mis-marshalled: {:?}", e))
        })?;

        let mut generation = String::new();
        if let Some(candidate) = resp.candidates.first() {
            if let Some(part) = candidate.content.parts.first() {
                generation = part.text.clone();
            }
        } else {
            return Err(LLMError::OtherError(
                "Expected at least one generated candidate from Gemini".to_string(),
            ));
        }

        let tokens = resp.usage_metadata.map(|u| TokenUsage {
            prompt_tokens: u.prompt_token_count,
            completion_tokens: u.candidates_token_count,
            total_tokens: u.total_token_count,
        });

        Ok(GenerateResult { tokens, generation })
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let req_body = self.build_request(messages);
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?alt=sse&key={}",
            self.model, self.api_key
        );

        let request_builder = self.client.post(&url).json(&req_body);

        let mut source = reqwest_eventsource::EventSource::new(request_builder).map_err(|e| {
            LLMError::OtherError(format!(
                "Cannot build stream connection request source: {:?}",
                e
            ))
        })?;

        let stream = async_stream::stream! {
            use reqwest_eventsource::Event;
            while let Some(event_res) = source.next().await {
                match event_res {
                    Ok(Event::Open) => continue,
                    Ok(Event::Message(message)) => {
                        let data_str = message.data.trim();
                        if data_str.is_empty() || data_str == "[DONE]" {
                            continue;
                        }

                        match serde_json::from_str::<GeminiResponse>(data_str) {
                            Ok(resp) => {
                                let mut text = String::new();
                                if let Some(candidate) = resp.candidates.first() {
                                    if let Some(part) = candidate.content.parts.first() {
                                        text = part.text.clone();
                                    }
                                } else {
                                    yield Err(LLMError::OtherError(format!("Malformed SSE chunk lacked a candidate segment: {}", data_str)));
                                    continue;
                                }

                                let val = serde_json::to_value(&resp).map_err(|e| LLMError::OtherError(format!("Failed generic chunk mapping back to Value: {:?}", e)));
                                match val {
                                    Ok(payload) => yield Ok(StreamData::new(payload, None, &text)),
                                    Err(e) => yield Err(e),
                                }
                            }
                            Err(e) => {
                                yield Err(LLMError::OtherError(format!("Failed to parse SSE JSON chunk: {} (data: {})", e, data_str)));
                            }
                        }
                    }
                    Err(e) => {
                        yield Err(LLMError::OtherError(format!("Event source error internally aborted transmission: {:?}", e)));
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    fn add_options(&mut self, options: CallOptions) {
        self.options.merge_options(options);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schemas::Message;

    #[test]
    fn test_message_mapping() {
        let gemini = Gemini::builder().api_key("test").build().unwrap();
        let messages = vec![
            Message::new_system_message("You are an expert"),
            Message::new_human_message("Hello"),
            Message::new_ai_message("Hi there"),
        ];

        let req = gemini.build_request(&messages);

        assert!(req.system_instruction.is_some());
        let sys_content = req.system_instruction.unwrap();
        assert_eq!(sys_content.parts[0].text, "You are an expert");

        assert_eq!(req.contents.len(), 2);
        assert_eq!(req.contents[0].role, "user");
        assert_eq!(req.contents[0].parts[0].text, "Hello");
        assert_eq!(req.contents[1].role, "model");
        assert_eq!(req.contents[1].parts[0].text, "Hi there");
    }

    #[test]
    fn test_request_serialization() {
        let gemini = Gemini::builder().api_key("test").build().unwrap();
        let messages = vec![
            Message::new_system_message("sys message"),
            Message::new_human_message("human message"),
        ];

        let req = gemini.build_request(&messages);
        let val = serde_json::to_value(&req).unwrap();

        let obj = val.as_object().unwrap();
        assert!(obj.contains_key("systemInstruction"));
        assert!(obj.contains_key("contents"));

        let contents = obj.get("contents").unwrap().as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
    }
}
