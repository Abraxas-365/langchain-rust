//! Audio and video transcripts from [AssemblyAI](https://www.assemblyai.com/docs) as
//! documents (langchaingo's `AssemblyAIAudioTranscriptLoader`).
//!
//! The loader submits the audio (a public URL, a local file or bytes, which are uploaded
//! first), polls until the transcript is ready and returns it as one document, one
//! document per sentence or paragraph, or SRT/VTT subtitles.
//!
//! ```no_run
//! use futures_util::StreamExt;
//! use langchain_rust::document_loaders::{AssemblyAILoader, AudioSource, Loader, TranscriptFormat};
//! use serde_json::json;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Reads ASSEMBLYAI_API_KEY.
//! let loader = AssemblyAILoader::new(AudioSource::url("https://assembly.ai/wildfires.mp3"))
//!     .with_format(TranscriptFormat::Paragraphs)
//!     .with_param("speaker_labels", json!(true));
//! let mut docs = loader.load().await?;
//! while let Some(doc) = docs.next().await {
//!     let doc = doc?;
//!     println!("[{} ms] {}", doc.metadata["start"], doc.page_content);
//! }
//! # Ok(())
//! # }
//! ```

use std::{collections::HashMap, path::PathBuf, pin::Pin, time::Duration};

use async_trait::async_trait;
use futures::{stream, Stream};
use serde_json::{json, Map, Value};

use crate::{
    document_loaders::{process_doc_stream, Loader, LoaderError},
    schemas::Document,
    text_splitter::TextSplitter,
};

pub const ASSEMBLYAI_API_BASE: &str = "https://api.assemblyai.com";
/// EU data residency endpoint, for [`AssemblyAILoader::with_base_url`].
pub const ASSEMBLYAI_EU_API_BASE: &str = "https://api.eu.assemblyai.com";
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_secs(3);
const DEFAULT_MAX_WAIT: Duration = Duration::from_secs(60 * 60);
const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const UPLOAD_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const MAX_ERROR_BODY_CHARS: usize = 500;

/// Where the audio comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioSource {
    /// A URL AssemblyAI can download (public, or presigned).
    Url(String),
    /// A local file, uploaded to AssemblyAI first.
    File(PathBuf),
    /// Raw audio bytes, uploaded to AssemblyAI first.
    Bytes(Vec<u8>),
    /// A transcript that was already submitted (for example from a webhook flow); it is
    /// polled and loaded without submitting anything.
    TranscriptId(String),
}

impl AudioSource {
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url(url.into())
    }

    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self::File(path.into())
    }

    pub fn transcript_id(id: impl Into<String>) -> Self {
        Self::TranscriptId(id.into())
    }

    /// The `source` metadata value. Bytes have no name, so it is `bytes`.
    fn label(&self) -> String {
        match self {
            Self::Url(url) => url.clone(),
            Self::File(path) => path.display().to_string(),
            Self::Bytes(_) => "bytes".to_string(),
            Self::TranscriptId(id) => format!("assemblyai:{id}"),
        }
    }
}

/// How the transcript is turned into documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TranscriptFormat {
    /// One document with the full text (default).
    #[default]
    Text,
    /// One document per sentence, with `start`/`end` (ms), `confidence` and `speaker`.
    Sentences,
    /// One document per paragraph, with the same metadata as sentences.
    Paragraphs,
    /// One document holding SRT subtitles.
    SubtitlesSrt,
    /// One document holding WebVTT subtitles.
    SubtitlesVtt,
}

/// Loads an AssemblyAI transcript as documents.
///
/// Every document gets `source` (the URL, file path, `bytes` or `assemblyai:<id>`) and
/// `transcript_id` metadata. With [`TranscriptFormat::Text`] the rest of the transcript
/// object is metadata too (`audio_duration`, `confidence`, `language_code`,
/// `utterances`, ...), except `text` (that is the page content) and the per-word
/// `words` array, which can be larger than the text itself. Sentence and paragraph
/// documents carry their own fields, also without `words`.
#[derive(Debug, Clone)]
pub struct AssemblyAILoader {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    source: AudioSource,
    format: TranscriptFormat,
    params: Map<String, Value>,
    chars_per_caption: Option<u32>,
    poll_interval: Duration,
    max_wait: Duration,
    request_timeout: Duration,
}

impl AssemblyAILoader {
    /// Reads the key from `ASSEMBLYAI_API_KEY` (loading fails with a clear error if it is
    /// unset; see [`with_api_key`](Self::with_api_key)).
    pub fn new(source: AudioSource) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: ASSEMBLYAI_API_BASE.to_string(),
            api_key: std::env::var("ASSEMBLYAI_API_KEY").unwrap_or_default(),
            source,
            format: TranscriptFormat::default(),
            params: Map::new(),
            chars_per_caption: None,
            poll_interval: DEFAULT_POLL_INTERVAL,
            max_wait: DEFAULT_MAX_WAIT,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// API origin: [`ASSEMBLYAI_API_BASE`] by default, [`ASSEMBLYAI_EU_API_BASE`] for EU
    /// data residency.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    pub fn with_format(mut self, format: TranscriptFormat) -> Self {
        self.format = format;
        self
    }

    /// Adds one transcription parameter to the submit request, such as
    /// `speaker_labels`, `language_code`, `speech_models` or `punctuate` (see
    /// AssemblyAI's "Transcribe audio" reference). `audio_url` is always set by the
    /// loader from the [`AudioSource`].
    pub fn with_param(mut self, key: impl Into<String>, value: Value) -> Self {
        self.params.insert(key.into(), value);
        self
    }

    /// Maximum characters per caption for subtitle formats.
    pub fn with_chars_per_caption(mut self, chars: u32) -> Self {
        self.chars_per_caption = Some(chars);
        self
    }

    /// How often to check whether the transcript is ready (default 3 s).
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Give up if the transcript is not ready after this long (default 1 hour). The
    /// transcript keeps processing on AssemblyAI's side; load it later with
    /// [`AudioSource::TranscriptId`] (the error message carries the id).
    pub fn with_max_wait(mut self, max_wait: Duration) -> Self {
        self.max_wait = max_wait;
        self
    }

    /// Timeout for each API call except uploads (default 60 s).
    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        self.client
            .request(method, format!("{}{path}", self.base_url))
            .header("authorization", &self.api_key)
            .timeout(self.request_timeout)
    }

    async fn upload(&self, bytes: Vec<u8>) -> Result<String, LoaderError> {
        if bytes.is_empty() {
            return Err(other("audio is empty"));
        }
        let response = self
            .request(reqwest::Method::POST, "/v2/upload")
            .timeout(UPLOAD_TIMEOUT)
            .header("content-type", "application/octet-stream")
            .body(bytes)
            .send()
            .await
            .map_err(http_error)?;
        let body = json_body(response, "upload").await?;
        body.get("upload_url")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| other("AssemblyAI upload response has no upload_url"))
    }

    async fn submit(&self, audio_url: String) -> Result<String, LoaderError> {
        let mut params = self.params.clone();
        params.insert("audio_url".into(), json!(audio_url));
        let response = self
            .request(reqwest::Method::POST, "/v2/transcript")
            .json(&params)
            .send()
            .await
            .map_err(http_error)?;
        let body = json_body(response, "submit").await?;
        let id = body
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| other("AssemblyAI submit response has no id"))?;
        check_id(id)?;
        Ok(id.to_string())
    }

    async fn wait_for(&self, id: &str) -> Result<Map<String, Value>, LoaderError> {
        let started = tokio::time::Instant::now();
        loop {
            let response = self
                .request(reqwest::Method::GET, &format!("/v2/transcript/{id}"))
                .send()
                .await
                .map_err(http_error)?;
            let transcript = match json_body(response, "get transcript").await? {
                Value::Object(map) => map,
                _ => return Err(other("AssemblyAI transcript is not a JSON object")),
            };
            match transcript.get("status").and_then(Value::as_str) {
                Some("completed") => return Ok(transcript),
                Some("error") => {
                    let reason = transcript
                        .get("error")
                        .and_then(Value::as_str)
                        .unwrap_or("no reason given");
                    return Err(other(format!(
                        "AssemblyAI transcript {id} failed: {reason}"
                    )));
                }
                // queued, processing, or a status added later: keep waiting.
                _ => {}
            }
            if started.elapsed() + self.poll_interval > self.max_wait {
                return Err(other(format!(
                    "AssemblyAI transcript {id} not ready after {:?}; load it later with \
                     AudioSource::TranscriptId",
                    self.max_wait
                )));
            }
            tokio::time::sleep(self.poll_interval).await;
        }
    }

    async fn transcript_id(&self) -> Result<String, LoaderError> {
        if self.api_key.is_empty() {
            return Err(other("ASSEMBLYAI_API_KEY is not set (or use with_api_key)"));
        }
        match &self.source {
            AudioSource::TranscriptId(id) => {
                check_id(id)?;
                Ok(id.clone())
            }
            AudioSource::Url(url) => {
                if url.trim().is_empty() {
                    return Err(other("audio URL is empty"));
                }
                self.submit(url.clone()).await
            }
            AudioSource::File(path) => {
                let bytes = tokio::fs::read(path).await?;
                let upload_url = self.upload(bytes).await?;
                self.submit(upload_url).await
            }
            AudioSource::Bytes(bytes) => {
                let upload_url = self.upload(bytes.clone()).await?;
                self.submit(upload_url).await
            }
        }
    }

    /// Transcribes (or fetches) and builds the documents.
    pub async fn load_documents(&self) -> Result<Vec<Document>, LoaderError> {
        let id = self.transcript_id().await?;
        let transcript = self.wait_for(&id).await?;
        let base = |extra: &[(&str, Value)]| {
            let mut metadata: HashMap<String, Value> = HashMap::from([
                ("source".to_string(), json!(self.source.label())),
                ("transcript_id".to_string(), json!(id)),
            ]);
            for (key, value) in extra {
                metadata.insert(key.to_string(), value.clone());
            }
            metadata
        };
        match self.format {
            TranscriptFormat::Text => {
                let text = transcript
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let mut metadata = base(&[]);
                for (key, value) in transcript {
                    if key != "text" && key != "words" && key != "id" {
                        metadata.insert(key, value);
                    }
                }
                Ok(vec![Document::new(text).with_metadata(metadata)])
            }
            TranscriptFormat::Sentences | TranscriptFormat::Paragraphs => {
                let field = if self.format == TranscriptFormat::Sentences {
                    "sentences"
                } else {
                    "paragraphs"
                };
                let response = self
                    .request(
                        reqwest::Method::GET,
                        &format!("/v2/transcript/{id}/{field}"),
                    )
                    .send()
                    .await
                    .map_err(http_error)?;
                let body = json_body(response, field).await?;
                let items = body
                    .get(field)
                    .and_then(Value::as_array)
                    .ok_or_else(|| other(format!("AssemblyAI response has no {field} array")))?;
                Ok(items
                    .iter()
                    .filter_map(Value::as_object)
                    .map(|item| {
                        let text = item
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let mut metadata = base(&[]);
                        for (key, value) in item {
                            if key != "text" && key != "words" {
                                metadata.insert(key.clone(), value.clone());
                            }
                        }
                        Document::new(text).with_metadata(metadata)
                    })
                    .collect())
            }
            TranscriptFormat::SubtitlesSrt | TranscriptFormat::SubtitlesVtt => {
                let kind = if self.format == TranscriptFormat::SubtitlesSrt {
                    "srt"
                } else {
                    "vtt"
                };
                let mut request =
                    self.request(reqwest::Method::GET, &format!("/v2/transcript/{id}/{kind}"));
                if let Some(chars) = self.chars_per_caption {
                    request = request.query(&[("chars_per_caption", chars)]);
                }
                let response = request.send().await.map_err(http_error)?;
                let response = check_status(response, kind).await?;
                let subtitles = response.text().await.map_err(http_error)?;
                Ok(vec![
                    Document::new(subtitles).with_metadata(base(&[("format", json!(kind))]))
                ])
            }
        }
    }
}

fn other(message: impl Into<String>) -> LoaderError {
    LoaderError::OtherError(message.into())
}

fn http_error(err: reqwest::Error) -> LoaderError {
    other(format!("AssemblyAI request failed: {err}"))
}

/// Transcript ids go into URL paths, so only plain id characters are accepted.
fn check_id(id: &str) -> Result<(), LoaderError> {
    let ok = !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if ok {
        Ok(())
    } else {
        Err(other(format!("invalid AssemblyAI transcript id {id:?}")))
    }
}

async fn check_status(
    response: reqwest::Response,
    what: &str,
) -> Result<reqwest::Response, LoaderError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let text = response.text().await.unwrap_or_default();
    let text: String = text.chars().take(MAX_ERROR_BODY_CHARS).collect();
    Err(other(format!(
        "AssemblyAI {what} failed ({status}): {text}"
    )))
}

async fn json_body(response: reqwest::Response, what: &str) -> Result<Value, LoaderError> {
    check_status(response, what)
        .await?
        .json()
        .await
        .map_err(|err| other(format!("AssemblyAI {what} returned invalid JSON: {err}")))
}

#[async_trait]
impl Loader for AssemblyAILoader {
    async fn load(
        self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let docs = self.load_documents().await?;
        Ok(Box::pin(stream::iter(docs.into_iter().map(Ok))))
    }

    async fn load_and_split<TS: TextSplitter + 'static>(
        self,
        splitter: TS,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let doc_stream = self.load().await?;
        let stream = process_doc_stream(doc_stream, splitter).await;
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
