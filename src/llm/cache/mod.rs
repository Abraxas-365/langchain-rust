//! Response caching for any [`LLM`], like langchaingo's `llms/cache`.

use std::{
    collections::HashMap,
    pin::Pin,
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use tokio::time::Instant;

use crate::{
    language_models::{
        box_llm_events, llm::LLM, options::CallOptions, result_events, GenerateResult, LLMError,
        LLMEvent, LLMEventStream,
    },
    schemas::{Message, StreamData},
};

/// Where [`CachedLLM`] keeps responses. Implement it to cache in Redis, a database, a
/// file, etc.
///
/// Keys are the exact JSON of the request (namespace, messages and call options), so
/// they never collide but can be long; hash them if your store needs short keys.
#[async_trait]
pub trait LLMCache: Send + Sync {
    /// The cached response for `key`, if any.
    async fn get(&self, key: &str) -> Option<GenerateResult>;
    /// Stores a response.
    async fn put(&self, key: &str, value: GenerateResult);
}

struct Entry {
    value: GenerateResult,
    inserted: Instant,
    last_used: u64,
}

#[derive(Default)]
struct InMemoryState {
    entries: HashMap<String, Entry>,
    clock: u64,
}

/// An in-process [`LLMCache`]. Unbounded by default; [`with_capacity`] evicts the least
/// recently used response and [`with_ttl`] expires responses after a while.
///
/// [`with_capacity`]: InMemoryLLMCache::with_capacity
/// [`with_ttl`]: InMemoryLLMCache::with_ttl
#[derive(Default)]
pub struct InMemoryLLMCache {
    state: Mutex<InMemoryState>,
    capacity: Option<usize>,
    ttl: Option<Duration>,
}

impl InMemoryLLMCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Keeps at most `capacity` responses, evicting the least recently used one.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = Some(capacity);
        self
    }

    /// Forgets responses older than `ttl`.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Number of stored responses (expired ones included until they are looked up or
    /// evicted).
    pub fn len(&self) -> usize {
        self.state.lock().unwrap().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&self) {
        self.state.lock().unwrap().entries.clear();
    }

    fn expired(&self, entry: &Entry, now: Instant) -> bool {
        self.ttl
            .is_some_and(|ttl| now.duration_since(entry.inserted) >= ttl)
    }
}

#[async_trait]
impl LLMCache for InMemoryLLMCache {
    async fn get(&self, key: &str) -> Option<GenerateResult> {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        state.clock += 1;
        let clock = state.clock;
        let expired = self.expired(state.entries.get(key)?, now);
        if expired {
            state.entries.remove(key);
            return None;
        }
        let entry = state.entries.get_mut(key)?;
        entry.last_used = clock;
        Some(entry.value.clone())
    }

    async fn put(&self, key: &str, value: GenerateResult) {
        let now = Instant::now();
        let mut state = self.state.lock().unwrap();
        state.clock += 1;
        let clock = state.clock;
        if self.capacity == Some(0) {
            return;
        }
        if let Some(capacity) = self.capacity {
            if !state.entries.contains_key(key) {
                // Expired entries go first, then the least recently used ones.
                state.entries.retain(|_, e| !self.expired(e, now));
                while state.entries.len() >= capacity {
                    let oldest = state
                        .entries
                        .iter()
                        .min_by_key(|(_, e)| e.last_used)
                        .map(|(k, _)| k.clone());
                    match oldest {
                        Some(k) => state.entries.remove(&k),
                        None => break,
                    };
                }
            }
        }
        state.entries.insert(
            key.to_string(),
            Entry {
                value,
                inserted: now,
                last_used: clock,
            },
        );
    }
}

/// Wraps an [`LLM`] and answers repeated requests from an [`LLMCache`] instead of calling
/// the model again.
///
/// The cache key is the namespace, the messages and the call options set through
/// [`add_options`](LLM::add_options) (temperature, stop words, tools, ...). The wrapper
/// can't see settings inside the model itself, such as the model name, so give each model
/// configuration its own [namespace](CachedLLM::with_namespace) when they share a cache.
/// By default the namespace is the wrapped type's name.
///
/// - A hit returns the stored result with `tokens: None`: it cost nothing. If a
///   `streaming_func` is set, it gets the whole text at once.
/// - Errors are never cached.
/// - [`generate_stream`](LLM::generate_stream) caches the final result of a streamed
///   response and replays hits as one `Text` event plus `Done`. The legacy
///   [`stream`](LLM::stream) replays hits but doesn't store misses.
/// - Concurrent identical misses each call the model.
///
/// ```rust,ignore
/// let cache = Arc::new(InMemoryLLMCache::new().with_ttl(Duration::from_secs(3600)));
/// let llm = CachedLLM::new(OpenAI::default(), cache).with_namespace("gpt-4o-mini");
/// let first = llm.invoke("Say hi").await?; // calls OpenAI
/// let again = llm.invoke("Say hi").await?; // from the cache
/// ```
pub struct CachedLLM {
    llm: Box<dyn LLM>,
    cache: Arc<dyn LLMCache>,
    namespace: String,
    options: CallOptions,
}

impl Clone for CachedLLM {
    fn clone(&self) -> Self {
        Self {
            llm: self.llm.clone_box(),
            cache: self.cache.clone(),
            namespace: self.namespace.clone(),
            options: self.options.clone(),
        }
    }
}

impl CachedLLM {
    pub fn new<L: LLM + 'static>(llm: L, cache: Arc<dyn LLMCache>) -> Self {
        Self {
            llm: Box::new(llm),
            cache,
            namespace: std::any::type_name::<L>().to_string(),
            options: CallOptions::new(),
        }
    }

    /// Separates this model's entries from others in a shared cache, e.g. the model name.
    pub fn with_namespace<S: Into<String>>(mut self, namespace: S) -> Self {
        self.namespace = namespace.into();
        self
    }

    /// The key a request is cached under.
    pub fn cache_key(&self, messages: &[Message]) -> Result<String, LLMError> {
        let key = json!({
            "namespace": self.namespace,
            "messages": messages,
            "options": options_fingerprint(&self.options),
        });
        Ok(serde_json::to_string(&key)?)
    }

    async fn lookup(&self, key: &str) -> Option<GenerateResult> {
        let mut hit = self.cache.get(key).await?;
        hit.tokens = None;
        Some(hit)
    }
}

/// The options that change what the model answers. `streaming_func` and
/// `stream_usage` only change how it's delivered.
fn options_fingerprint(o: &CallOptions) -> Value {
    json!({
        "candidate_count": o.candidate_count,
        "max_tokens": o.max_tokens,
        "temperature": o.temperature,
        "stop_words": o.stop_words,
        "top_k": o.top_k,
        "top_p": o.top_p,
        "seed": o.seed,
        "min_length": o.min_length,
        "max_length": o.max_length,
        "n": o.n,
        "repetition_penalty": o.repetition_penalty,
        "frequency_penalty": o.frequency_penalty,
        "presence_penalty": o.presence_penalty,
        "functions": o.functions.as_ref().map(|fs| {
            fs.iter()
                .map(|f| json!([f.name, f.description, f.parameters]))
                .collect::<Vec<_>>()
        }),
        "function_call_behavior": o.function_call_behavior.as_ref().map(|b| format!("{b:?}")),
        "response_format": o.response_format.as_ref().map(|r| format!("{r:?}")),
    })
}

#[async_trait]
impl LLM for CachedLLM {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let key = self.cache_key(messages)?;
        if let Some(hit) = self.lookup(&key).await {
            if let Some(func) = &self.options.streaming_func {
                let mut func = func.lock().await;
                func(hit.generation.clone())
                    .await
                    .map_err(|_| LLMError::OtherError("streaming_func failed".into()))?;
            }
            return Ok(hit);
        }
        let result = self.llm.generate(messages).await?;
        self.cache.put(&key, result.clone()).await;
        Ok(result)
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let key = self.cache_key(messages)?;
        if let Some(hit) = self.lookup(&key).await {
            let chunk = StreamData::new(json!(hit.generation), None, hit.generation);
            return Ok(Box::pin(futures::stream::iter([Ok(chunk)])));
        }
        self.llm.stream(messages).await
    }

    fn generate_stream<'a>(&'a self, messages: &'a [Message]) -> LLMEventStream<'a> {
        box_llm_events(async_stream::try_stream! {
            let key = self.cache_key(messages)?;
            if let Some(hit) = self.lookup(&key).await {
                for event in result_events(hit) {
                    yield event;
                }
                return;
            }
            let mut events = self.llm.generate_stream(messages);
            while let Some(event) = events.next().await {
                let event = event?;
                if let LLMEvent::Done(result) = &event {
                    self.cache.put(&key, result.clone()).await;
                }
                yield event;
            }
        })
    }

    fn add_options(&mut self, options: CallOptions) {
        self.options.merge_options(options.clone());
        self.llm.add_options(options);
    }

    fn messages_to_string(&self, messages: &[Message]) -> String {
        self.llm.messages_to_string(messages)
    }
}

#[cfg(test)]
mod tests;
