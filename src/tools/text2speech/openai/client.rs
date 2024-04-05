use std::{error::Error, sync::Arc};

use async_openai::{
    config::{Config, OpenAIConfig},
    types::{CreateSpeechRequestArgs, SpeechModel, SpeechResponseFormat, Voice},
    Client,
};
use async_trait::async_trait;
use serde_json::Value;

use crate::tools::{SpeechStorage, Tool};

use super::models::{OpenAIVoices, OpenAiResponseFormat, Text2SpeechOpenAIModel};

#[derive(Clone)]
pub struct Text2SpeechOpenAI<C: Config> {
    config: C,
    model: String,
    voice: String,
    storage: Option<Arc<dyn SpeechStorage>>,
    response_format: OpenAiResponseFormat,
    path: String,
}

impl<C: Config> Text2SpeechOpenAI<C> {
    pub fn new(config: C) -> Self {
        Self {
            config,
            model: Text2SpeechOpenAIModel::TTS1.into(),
            voice: OpenAIVoices::Alloy.into(),
            storage: None,
            response_format: OpenAiResponseFormat::Mp3,
            path: "./data/audio.mp3".to_string(),
        }
    }

    pub fn wiith_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_voice<S: Into<String>>(mut self, voice: S) -> Self {
        self.voice = voice.into();
        self
    }

    pub fn with_storage<SS: SpeechStorage + 'static>(mut self, storage: SS) -> Self {
        self.storage = Some(Arc::new(storage));
        self
    }

    pub fn with_response_format(mut self, response_format: OpenAiResponseFormat) -> Self {
        self.response_format = response_format;
        self
    }

    pub fn with_path<S: Into<String>>(mut self, path: S) -> Self {
        self.path = path.into();
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
        format!(
            r#"A wrapper around OpenAI Text2Speech. "
        "Useful for when you need to convert text to speech. "
        "It supports multiple languages, including English, German, Polish, "
        "Spanish, Italian, French, Portuguese""#
        )
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let input = input.as_str().ok_or("Invalid input")?;
        let client = Client::new();
        let response_format: SpeechResponseFormat = self.response_format.clone().into();

        let request = CreateSpeechRequestArgs::default()
            .input(input)
            .voice(Voice::Other(self.voice.clone()))
            .response_format(response_format)
            .model(SpeechModel::Other(self.model.clone()))
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
    use crate::tools::{Text2SpeechOpenAI, Tool};

    #[tokio::test]
    #[ignore]
    async fn openai_speech2text_tool() {
        let openai = Text2SpeechOpenAI::default();
        let s = openai.call("Hola como estas").await.unwrap();
        println!("{}", s);
    }
}
