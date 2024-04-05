use async_openai::types::{SpeechResponseFormat, Voice};

#[derive(Clone)]
pub enum Text2SpeechOpenAIModel {
    TTS1,
    TTS1HD,
}

impl Into<String> for Text2SpeechOpenAIModel {
    fn into(self) -> String {
        match self {
            Self::TTS1 => "tts-1".to_string(),
            Self::TTS1HD => "tts-1-hd".to_string(),
        }
    }
}

pub enum OpenAIVoices {
    Alloy,
    Echo,
    Fable,
    Onyx,
    Nova,
    Shimmer,
}
impl Into<String> for OpenAIVoices {
    fn into(self) -> String {
        match self {
            Self::Alloy => "alloy".to_string(),
            Self::Echo => "echo".to_string(),
            Self::Fable => "fable".to_string(),
            Self::Onyx => "onyx".to_string(),
            Self::Nova => "nova".to_string(),
            Self::Shimmer => "shimmer".to_string(),
        }
    }
}

impl Into<Voice> for OpenAIVoices {
    fn into(self) -> Voice {
        match self {
            Self::Alloy => Voice::Alloy,
            Self::Echo => Voice::Echo,
            Self::Fable => Voice::Fable,
            Self::Onyx => Voice::Onyx,
            Self::Nova => Voice::Nova,
            Self::Shimmer => Voice::Shimmer,
        }
    }
}

#[derive(Clone)]
pub enum OpenAiResponseFormat {
    Mp3,
    Opus,
    Aac,
    Flac,
    Pcm,
    Wav,
}

impl Into<SpeechResponseFormat> for OpenAiResponseFormat {
    fn into(self) -> SpeechResponseFormat {
        match self {
            Self::Mp3 => SpeechResponseFormat::Mp3,
            Self::Opus => SpeechResponseFormat::Opus,
            Self::Aac => SpeechResponseFormat::Aac,
            Self::Flac => SpeechResponseFormat::Flac,
            Self::Pcm => SpeechResponseFormat::Pcm,
            Self::Wav => SpeechResponseFormat::Wav,
        }
    }
}
