use std::{sync::atomic::AtomicUsize, sync::atomic::Ordering};

use futures::StreamExt;
use serde_json::json;

use super::*;
use crate::{
    chain::{Chain, LLMChainBuilder},
    language_models::TokenUsage,
    prompt_args,
    schemas::{FunctionCallBehavior, FunctionDefinition, ResponseFormat},
    template_jinja2,
    test_utils::FakeLLM,
};

fn wrap(llm: &FakeLLM) -> (CachedLLM, Arc<InMemoryLLMCache>) {
    let cache = Arc::new(InMemoryLLMCache::new());
    (CachedLLM::new(llm.clone(), cache.clone()), cache)
}

fn result(text: &str) -> GenerateResult {
    GenerateResult {
        generation: text.into(),
        ..Default::default()
    }
}

#[tokio::test]
async fn repeated_requests_hit_the_cache() {
    let llm = FakeLLM::new(["first", "second"]);
    let (cached, cache) = wrap(&llm);

    let miss = cached
        .generate(&[Message::new_human_message("hi")])
        .await
        .unwrap();
    assert_eq!(miss.generation, "first");
    assert_eq!(miss.tokens, Some(TokenUsage::new(1, 1)));

    // Same request: no model call, no token cost.
    let hit = cached
        .generate(&[Message::new_human_message("hi")])
        .await
        .unwrap();
    assert_eq!(hit.generation, "first");
    assert_eq!(hit.tokens, None);
    assert_eq!(llm.calls().len(), 1);
    // The stored entry keeps its usage; only the returned copy drops it.
    let key = cached
        .cache_key(&[Message::new_human_message("hi")])
        .unwrap();
    assert_eq!(
        cache.get(&key).await.unwrap().tokens,
        Some(TokenUsage::new(1, 1))
    );

    // Different messages (content or role) are different requests.
    let other = cached.invoke("bye").await.unwrap();
    assert_eq!(other, "second");
    let err = cached
        .generate(&[Message::new_system_message("hi")])
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("no scripted response left"),
        "{err}"
    );
    assert_eq!(llm.calls().len(), 3);
    assert_eq!(cache.len(), 2);
}

#[tokio::test]
async fn errors_are_not_cached() {
    let llm = FakeLLM::with_results([]);
    let (cached, cache) = wrap(&llm);
    assert!(cached.invoke("hi").await.is_err());
    assert!(cache.is_empty());
}

#[tokio::test]
async fn options_are_part_of_the_key() {
    let llm = FakeLLM::new(["a", "b", "c", "d"]);
    let (mut cached, _) = wrap(&llm);
    let msgs = [Message::new_human_message("hi")];

    let base = cached.cache_key(&msgs).unwrap();
    cached.add_options(CallOptions::new().with_temperature(0.2));
    let warm = cached.cache_key(&msgs).unwrap();
    assert_ne!(base, warm);
    // The wrapped model gets the options too.
    let forwarded = llm.options.lock().unwrap().clone();
    assert_eq!(forwarded.len(), 1);
    assert_eq!(forwarded[0].temperature, Some(0.2));

    let mut keys = vec![base, warm.clone()];
    for opts in [
        CallOptions::new().with_max_tokens(10),
        CallOptions::new().with_stop_words(vec!["X".into()]),
        CallOptions::new().with_seed(7),
        CallOptions::new().with_top_p(0.5),
        CallOptions::new().with_functions(vec![FunctionDefinition::new(
            "f",
            "does f",
            json!({"type": "object"}),
        )]),
        CallOptions::new().with_function_call_behavior(FunctionCallBehavior::Auto),
        CallOptions::new().with_response_format(ResponseFormat::JsonObject),
    ] {
        cached.add_options(opts);
        keys.push(cached.cache_key(&msgs).unwrap());
    }
    let unique: std::collections::HashSet<_> = keys.iter().collect();
    assert_eq!(unique.len(), keys.len(), "every option changes the key");

    // Delivery-only options don't.
    let before = cached.cache_key(&msgs).unwrap();
    cached.add_options(CallOptions::new().with_stream_usage(true));
    assert_eq!(cached.cache_key(&msgs).unwrap(), before);

    // Namespaces separate entries.
    let a = cached
        .clone()
        .with_namespace("model-a")
        .cache_key(&msgs)
        .unwrap();
    let b = cached
        .clone()
        .with_namespace("model-b")
        .cache_key(&msgs)
        .unwrap();
    assert_ne!(a, b);
    assert!(
        before.contains("FakeLLM"),
        "default namespace is the type: {before}"
    );
}

#[tokio::test]
async fn shared_cache_across_wrappers_and_clones() {
    let cache = Arc::new(InMemoryLLMCache::new());
    let llm_a = FakeLLM::new(["from a"]);
    let llm_b = FakeLLM::new(["from b"]);
    let a = CachedLLM::new(llm_a.clone(), cache.clone()).with_namespace("m");
    let b = CachedLLM::new(llm_b.clone(), cache.clone()).with_namespace("m");
    assert_eq!(a.invoke("q").await.unwrap(), "from a");
    // Same namespace and request: b is served from a's entry.
    assert_eq!(b.invoke("q").await.unwrap(), "from a");
    assert!(llm_b.calls().is_empty());
    // A clone shares the cache.
    assert_eq!(a.clone().invoke("q").await.unwrap(), "from a");
    assert_eq!(llm_a.calls().len(), 1);
}

#[tokio::test]
async fn hits_feed_streaming_func() {
    let llm = FakeLLM::new(["hello world"]);
    let (mut cached, _) = wrap(&llm);
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = seen.clone();
    cached.add_options(
        CallOptions::new().with_streaming_func(move |chunk: String| {
            sink.lock().unwrap().push(chunk);
            async { Ok(()) }
        }),
    );
    cached.invoke("hi").await.unwrap();
    assert!(
        seen.lock().unwrap().is_empty(),
        "the miss streams through the model"
    );
    cached.invoke("hi").await.unwrap();
    assert_eq!(*seen.lock().unwrap(), ["hello world"]);

    let llm = FakeLLM::new(["x"]);
    let (mut cached, _) = wrap(&llm);
    cached.add_options(CallOptions::new().with_streaming_func(|_: String| async { Err(()) }));
    cached.invoke("hi").await.unwrap();
    let err = cached.invoke("hi").await.unwrap_err();
    assert!(err.to_string().contains("streaming_func"), "{err}");
}

#[tokio::test]
async fn generate_stream_caches_and_replays() {
    let llm = FakeLLM::new(["one two three"]);
    let (cached, cache) = wrap(&llm);
    let msgs = [Message::new_human_message("count")];

    let collect = |cached: CachedLLM| {
        let msgs = msgs.clone();
        async move {
            cached
                .generate_stream(&msgs)
                .map(|e| match e.unwrap() {
                    LLMEvent::Text(t) => format!("T:{t}"),
                    LLMEvent::Done(r) => format!("D:{}:{:?}", r.generation, r.tokens),
                })
                .collect::<Vec<_>>()
                .await
        }
    };
    // The miss streams chunk by chunk and stores the final result.
    assert_eq!(
        collect(cached.clone()).await,
        [
            "T:one ",
            "T:two ",
            "T:three",
            "D:one two three:Some(TokenUsage { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 })"
        ]
    );
    assert_eq!(cache.len(), 1);
    // The hit replays in one piece, free.
    assert_eq!(
        collect(cached.clone()).await,
        ["T:one two three", "D:one two three:None"]
    );
    assert_eq!(llm.calls().len(), 1);
    // generate() shares the entry.
    assert_eq!(
        cached.generate(&msgs).await.unwrap().generation,
        "one two three"
    );

    // A failed stream stores nothing.
    let llm = FakeLLM::with_results([]);
    let (cached, cache) = wrap(&llm);
    let events: Vec<_> = cached.generate_stream(&msgs).collect().await;
    assert!(events[0].is_err());
    assert!(cache.is_empty());
}

#[tokio::test]
async fn legacy_stream_replays_hits() {
    let llm = FakeLLM::new(["a b", "c d"]);
    let (cached, _) = wrap(&llm);
    let msgs = [Message::new_human_message("q")];
    let text = |cached: CachedLLM| {
        let msgs = msgs.clone();
        async move {
            let chunks: Vec<_> = cached.stream(&msgs).await.unwrap().collect().await;
            chunks
                .into_iter()
                .map(|c| c.unwrap().content)
                .collect::<Vec<_>>()
        }
    };
    // Not cached yet: streams from the model and stores nothing.
    assert_eq!(text(cached.clone()).await, ["a ", "b"]);
    assert_eq!(text(cached.clone()).await, ["c ", "d"]);
    cached.generate(&msgs).await.unwrap_err();
    // Once generate() stored it, stream() replays it.
    let llm = FakeLLM::new(["e f"]);
    let (cached, _) = wrap(&llm);
    cached.generate(&msgs).await.unwrap();
    assert_eq!(text(cached.clone()).await, ["e f"]);
    assert_eq!(llm.calls().len(), 1);
}

#[tokio::test]
async fn works_inside_a_chain() {
    let llm = FakeLLM::new(["Paris"]);
    let (cached, _) = wrap(&llm);
    let chain = LLMChainBuilder::new()
        .prompt(template_jinja2!("Capital of {{country}}?", "country"))
        .llm(cached)
        .build()
        .unwrap();
    for _ in 0..3 {
        assert_eq!(
            chain
                .invoke(prompt_args! {"country" => "France"})
                .await
                .unwrap(),
            "Paris"
        );
    }
    assert_eq!(llm.calls().len(), 1);
}

#[tokio::test]
async fn in_memory_basic() {
    let cache = InMemoryLLMCache::new();
    assert!(cache.get("k").await.is_none());
    cache.put("k", result("v")).await;
    cache.put("k", result("v2")).await;
    assert_eq!(cache.get("k").await.unwrap().generation, "v2");
    assert_eq!(cache.len(), 1);
    cache.clear();
    assert!(cache.is_empty() && cache.get("k").await.is_none());
}

#[tokio::test]
async fn in_memory_lru_capacity() {
    let cache = InMemoryLLMCache::new().with_capacity(2);
    cache.put("a", result("A")).await;
    cache.put("b", result("B")).await;
    // Touch a, so b is the least recently used.
    assert!(cache.get("a").await.is_some());
    cache.put("c", result("C")).await;
    assert_eq!(cache.len(), 2);
    assert!(cache.get("b").await.is_none());
    assert_eq!(cache.get("a").await.unwrap().generation, "A");
    assert_eq!(cache.get("c").await.unwrap().generation, "C");
    // Overwriting a present key evicts nothing, even when it isn't the oldest one
    // (here `a` is the least recently used).
    cache.put("c", result("C2")).await;
    assert_eq!(cache.len(), 2);
    assert_eq!(cache.get("a").await.unwrap().generation, "A");
    assert_eq!(cache.get("c").await.unwrap().generation, "C2");

    let none = InMemoryLLMCache::new().with_capacity(0);
    none.put("a", result("A")).await;
    assert!(none.is_empty());
}

#[tokio::test(start_paused = true)]
async fn in_memory_ttl() {
    let cache = InMemoryLLMCache::new().with_ttl(Duration::from_secs(10));
    cache.put("a", result("A")).await;
    tokio::time::advance(Duration::from_secs(9)).await;
    assert!(cache.get("a").await.is_some(), "still fresh");
    tokio::time::advance(Duration::from_secs(1)).await;
    assert!(cache.get("a").await.is_none(), "expired at the TTL");
    assert!(cache.is_empty(), "expired entries are dropped on lookup");

    // A full cache drops expired entries before evicting live ones.
    let cache = InMemoryLLMCache::new()
        .with_capacity(2)
        .with_ttl(Duration::from_secs(10));
    cache.put("old", result("O")).await;
    tokio::time::advance(Duration::from_secs(5)).await;
    cache.put("live", result("L")).await;
    assert!(
        cache.get("old").await.is_some(),
        "old is now the most recent"
    );
    tokio::time::advance(Duration::from_secs(5)).await;
    cache.put("new", result("N")).await;
    assert!(
        cache.get("live").await.is_some(),
        "live survived; expired old went"
    );
    assert!(cache.get("new").await.is_some());
}

/// A custom backend, as a user would write one.
#[tokio::test]
async fn custom_backend() {
    #[derive(Default)]
    struct Counting {
        inner: InMemoryLLMCache,
        gets: AtomicUsize,
        puts: AtomicUsize,
    }
    #[async_trait]
    impl LLMCache for Counting {
        async fn get(&self, key: &str) -> Option<GenerateResult> {
            self.gets.fetch_add(1, Ordering::SeqCst);
            self.inner.get(key).await
        }
        async fn put(&self, key: &str, value: GenerateResult) {
            self.puts.fetch_add(1, Ordering::SeqCst);
            self.inner.put(key, value).await
        }
    }
    let backend = Arc::new(Counting::default());
    let cached = CachedLLM::new(FakeLLM::new(["x"]), backend.clone());
    cached.invoke("q").await.unwrap();
    cached.invoke("q").await.unwrap();
    assert_eq!(backend.gets.load(Ordering::SeqCst), 2);
    assert_eq!(backend.puts.load(Ordering::SeqCst), 1);
}

/// Live: `OLLAMA_MODEL=qwen3:14b cargo test --lib llm::cache::tests::live -- --ignored --nocapture`
#[tokio::test]
#[ignore]
async fn live_ollama_cache() {
    use crate::llm::{ollama::openai::OllamaConfig, openai::OpenAI};
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:14b".into());
    let cached = CachedLLM::new(
        OpenAI::new(OllamaConfig::default()).with_model(&model),
        Arc::new(InMemoryLLMCache::new()),
    )
    .with_namespace(model);
    let msgs = [Message::new_human_message(
        "Write one short sentence about the sea.",
    )];

    let start = std::time::Instant::now();
    let first = cached.generate(&msgs).await.unwrap();
    let miss = start.elapsed();
    let start = std::time::Instant::now();
    let second = cached.generate(&msgs).await.unwrap();
    let hit = start.elapsed();
    println!("miss {miss:?} {:?}\nhit {hit:?}", first.tokens);
    assert_eq!(first.generation, second.generation);
    assert!(first.tokens.is_some() && second.tokens.is_none());
    assert!(hit < miss / 10, "hit {hit:?} vs miss {miss:?}");
}
