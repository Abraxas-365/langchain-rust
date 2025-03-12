use std::{error::Error, sync::Arc};

use async_openai::types::CreateSpeechRequestArgs;
use async_openai::Client;
pub use async_openai::{
    config::{Config, OpenAIConfig},
    types::{SpeechModel, SpeechResponseFormat, Voice},
};
use async_trait::async_trait;
use serde_json::Value;

use crate::tools::{SpeechStorage, Tool};

#[derive(Clone)]
pub struct Text2SpeechOpenAI<C: Config> {
    config: C,
    model: SpeechModel,
    voice: Voice,
    storage: Option<Arc<dyn SpeechStorage>>,
    response_format: SpeechResponseFormat,
    path: String,
}

impl<C: Config> Text2SpeechOpenAI<C> {
    pub fn new(config: C) -> Self {
        Self {
            config,
            model: SpeechModel::Tts1,
            voice: Voice::Alloy,
            storage: None,
            response_format: SpeechResponseFormat::Mp3,
            path: "./data/audio.mp3".to_string(),
        }
    }

    pub fn with_model(mut self, model: SpeechModel) -> Self {
        self.model = model;
        self
    }

    pub fn with_voice(mut self, voice: Voice) -> Self {
        self.voice = voice;
        self
    }

    pub fn with_storage<SS: SpeechStorage + 'static>(mut self, storage: SS) -> Self {
        self.storage = Some(Arc::new(storage));
        self
    }

    pub fn with_response_format(mut self, response_format: SpeechResponseFormat) -> Self {
        self.response_format = response_format;
        self
    }

    pub fn with_path<S: Into<String>>(mut self, path: S) -> Self {
        self.path = path.into();
        self
    }

    pub fn with_config(mut self, config: C) -> Self {
        self.config = config;
        self
    }
}

impl Default for Text2SpeechOpenAI<OpenAIConfig> {
    fn default() -> Self {
        Self::new(OpenAIConfig::default())
    }
}

#[async_trait]
impl<C: Config + Send + Sync> Tool for Text2SpeechOpenAI<C> {
    fn name(&self) -> String {
        "Text2SpeechOpenAI".to_string()
    }

    fn description(&self) -> String {
        r#"A wrapper around OpenAI Text2Speech. "
        "Useful for when you need to convert text to speech. "
        "It supports multiple languages, including English, German, Polish, "
        "Spanish, Italian, French, Portuguese""#
            .to_string()
    }

    async fn call(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let input = input.as_str().ok_or("Invalid input")?;
        let client = Client::new();
        let response_format: SpeechResponseFormat = self.response_format;

        let request = CreateSpeechRequestArgs::default()
            .input(input)
            .voice(self.voice.clone())
            .response_format(response_format)
            .model(self.model.clone())
            .build()?;

        let response = client.audio().speech(request).await?;

        if self.storage.is_some() {
            let storage = self.storage.as_ref().unwrap(); //safe to unwrap
            let data = response.bytes;
            return storage.save(&self.path, &data).await;
        } else {
            response.save(&self.path).await?;
        }

        Ok(self.path.clone())
    }
}

#[cfg(test)]
mod tests {
    use serde_json::Value;

    use crate::tools::{Text2SpeechOpenAI, Tool};

    #[tokio::test]
    #[ignore]
    async fn openai_speech2text_tool() {
        let openai = Text2SpeechOpenAI::default();
        let s = openai
            .call(Value::String("Hola como estas".to_string()))
            .await
            .unwrap();
        println!("{}", s);
    }
}
