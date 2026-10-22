//! Offline test doubles shared by unit tests. Compiled only for `cargo test`.
use std::{
    collections::{HashMap, VecDeque},
    pin::Pin,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use futures::{stream, Stream};

use crate::{
    embedding::{Embedder, EmbedderError},
    language_models::{llm::LLM, options::CallOptions, GenerateResult, LLMError, TokenUsage},
    schemas::{Message, StreamData},
};

/// LLM that replays scripted responses in order and records every prompt it receives.
#[derive(Clone, Default)]
pub struct FakeLLM {
    responses: Arc<Mutex<VecDeque<GenerateResult>>>,
    pub calls: Arc<Mutex<Vec<Vec<Message>>>>,
    pub options: Arc<Mutex<Vec<CallOptions>>>,
}

impl FakeLLM {
    pub fn new<S: Into<String>>(responses: impl IntoIterator<Item = S>) -> Self {
        Self::with_results(responses.into_iter().map(|s| GenerateResult {
            generation: s.into(),
            tokens: Some(TokenUsage::new(1, 1)),
            ..Default::default()
        }))
    }

    /// Scripts full results, e.g. typed `tool_calls` alongside plain text.
    pub fn with_results(results: impl IntoIterator<Item = GenerateResult>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(results.into_iter().collect())),
            ..Default::default()
        }
    }

    pub fn calls(&self) -> Vec<Vec<Message>> {
        self.calls.lock().unwrap().clone()
    }

    fn next(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        self.calls.lock().unwrap().push(messages.to_vec());
        self.responses
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| LLMError::OtherError("FakeLLM: no scripted response left".into()))
    }
}

#[async_trait]
impl LLM for FakeLLM {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        self.next(messages)
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let text = self.next(messages)?.generation;
        let chunks: Vec<Result<StreamData, LLMError>> = text
            .split_inclusive(' ')
            .map(|c| Ok(StreamData::new(serde_json::json!(c), None, c)))
            .collect();
        Ok(Box::pin(stream::iter(chunks)))
    }

    fn add_options(&mut self, options: CallOptions) {
        self.options.lock().unwrap().push(options);
    }
}

/// Embedder that maps known texts to fixed vectors; unknown texts get a zero-ish vector.
#[derive(Clone, Default)]
pub struct FakeEmbedder {
    pub vectors: HashMap<String, Vec<f64>>,
    pub dim: usize,
}

impl FakeEmbedder {
    pub fn new(dim: usize, entries: &[(&str, Vec<f64>)]) -> Self {
        Self {
            vectors: entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
            dim,
        }
    }

    fn lookup(&self, text: &str) -> Vec<f64> {
        self.vectors.get(text).cloned().unwrap_or_else(|| {
            let mut v = vec![0.0; self.dim];
            if let Some(last) = v.last_mut() {
                *last = 1.0;
            }
            v
        })
    }
}

#[async_trait]
impl Embedder for FakeEmbedder {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        Ok(documents.iter().map(|d| self.lookup(d)).collect())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        Ok(self.lookup(text))
    }
}
