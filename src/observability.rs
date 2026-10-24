//! `tracing` instrumentation shared by providers, chains and agents.
//!
//! Span names and fields are part of the public surface (people build dashboards on them):
//!
//! | span | fields |
//! |---|---|
//! | `llm.generate` | `provider`, `model`, `stream`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `cache_read_tokens`, `cache_creation_tokens`, `tool_calls`, `error` |
//! | `embed.documents` / `embed.query` | `provider`, `model`, `count`, `error` |
//! | `chain.call` | `chain`, `error` |
//! | `agent.run` | `max_iterations`, `steps`, `total_tokens`, `error` |
//! | `agent.plan` | `step` |
//! | `tool.call` | `tool`, `error` |
//!
//! Prompts and outputs are never recorded on spans; `debug!` events carry them for those
//! who opt in through their subscriber filter.
use std::{
    fmt::Display,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use futures::Stream;
use tracing::{field::Empty, Instrument, Span};

use crate::language_models::{GenerateResult, LLMError, LLMEvent, LLMEventStream, TokenUsage};

pub(crate) fn llm_span(provider: &'static str, model: &str, stream: bool) -> Span {
    tracing::info_span!(
        "llm.generate",
        provider,
        model,
        stream,
        prompt_tokens = Empty,
        completion_tokens = Empty,
        total_tokens = Empty,
        cache_read_tokens = Empty,
        cache_creation_tokens = Empty,
        tool_calls = Empty,
        error = Empty,
    )
}

fn record_result(span: &Span, result: &GenerateResult) {
    if let Some(TokenUsage {
        prompt_tokens,
        completion_tokens,
        total_tokens,
        cache_read_tokens,
        cache_creation_tokens,
    }) = &result.tokens
    {
        span.record("prompt_tokens", prompt_tokens);
        span.record("completion_tokens", completion_tokens);
        span.record("total_tokens", total_tokens);
        // Only when the provider reports cache activity, to keep spans small.
        if *cache_read_tokens > 0 || *cache_creation_tokens > 0 {
            span.record("cache_read_tokens", cache_read_tokens);
            span.record("cache_creation_tokens", cache_creation_tokens);
        }
    }
    if result.has_tool_calls() {
        span.record("tool_calls", result.tool_calls.len());
    }
}

/// Records `err` on `span` and logs it as an event inside the span.
pub(crate) fn record_error(span: &Span, err: &impl Display) {
    span.record("error", tracing::field::display(err));
    span.in_scope(|| tracing::warn!(error = %err, "{} failed", span_name(span)));
}

fn span_name(span: &Span) -> &'static str {
    span.metadata().map(|m| m.name()).unwrap_or("operation")
}

/// Runs an LLM call inside an `llm.generate` span, recording usage or the error.
pub(crate) async fn traced_generate(
    provider: &'static str,
    model: &str,
    call: impl Future<Output = Result<GenerateResult, LLMError>>,
) -> Result<GenerateResult, LLMError> {
    let span = llm_span(provider, model, false);
    traced(span, call, record_result).await
}

/// Runs `fut` inside `span`; on success `on_ok` records fields, on error the error is recorded.
pub(crate) async fn traced<T, E: Display>(
    span: Span,
    fut: impl Future<Output = Result<T, E>>,
    on_ok: impl FnOnce(&Span, &T),
) -> Result<T, E> {
    let result = fut.instrument(span.clone()).await;
    match &result {
        Ok(value) => on_ok(&span, value),
        Err(err) => record_error(&span, err),
    }
    result
}

/// Wraps a [`LLMEventStream`] in an `llm.generate` span (`stream = true`). The span covers
/// every poll and closes when the stream is dropped.
pub(crate) fn traced_event_stream<'a>(
    provider: &'static str,
    model: &str,
    stream: LLMEventStream<'a>,
) -> LLMEventStream<'a> {
    Box::pin(TracedEvents {
        inner: stream,
        span: llm_span(provider, model, true),
    })
}

struct TracedEvents<'a> {
    inner: LLMEventStream<'a>,
    span: Span,
}

impl Stream for TracedEvents<'_> {
    type Item = Result<LLMEvent, LLMError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        let poll = {
            let _entered = this.span.enter();
            this.inner.as_mut().poll_next(cx)
        };
        match &poll {
            Poll::Ready(Some(Ok(LLMEvent::Done(result)))) => record_result(&this.span, result),
            Poll::Ready(Some(Err(err))) => record_error(&this.span, err),
            _ => {}
        }
        poll
    }
}

/// Runs `embed_documents` inside an `embed.documents` span (`count` = number of documents).
pub(crate) async fn traced_embed_documents<T, E: Display>(
    provider: &'static str,
    model: &str,
    count: usize,
    fut: impl Future<Output = Result<T, E>>,
) -> Result<T, E> {
    let span = tracing::info_span!("embed.documents", provider, model, count, error = Empty);
    traced(span, fut, |_, _| {}).await
}

/// Runs `embed_query` inside an `embed.query` span.
pub(crate) async fn traced_embed_query<T, E: Display>(
    provider: &'static str,
    model: &str,
    fut: impl Future<Output = Result<T, E>>,
) -> Result<T, E> {
    let span = tracing::info_span!("embed.query", provider, model, count = 1, error = Empty);
    traced(span, fut, |_, _| {}).await
}

#[cfg(test)]
pub(crate) mod testing {
    //! A `tracing` layer that captures closed spans with their fields, for assertions.
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
    };

    use tracing::{
        field::{Field, Visit},
        span::{Attributes, Id, Record},
        Subscriber,
    };
    use tracing_subscriber::{layer::Context, prelude::*, registry::LookupSpan, Layer};

    #[derive(Debug, Clone)]
    pub struct SpanRecord {
        pub name: &'static str,
        pub fields: HashMap<String, String>,
        pub parent: Option<&'static str>,
    }

    impl SpanRecord {
        pub fn field(&self, name: &str) -> Option<&str> {
            self.fields.get(name).map(String::as_str)
        }
    }

    #[derive(Clone, Default)]
    pub struct Captured(Arc<Mutex<Vec<SpanRecord>>>);

    impl Captured {
        /// Closed spans, in closing order.
        pub fn spans(&self) -> Vec<SpanRecord> {
            self.0.lock().unwrap().clone()
        }

        pub fn named(&self, name: &str) -> Vec<SpanRecord> {
            self.spans()
                .into_iter()
                .filter(|s| s.name == name)
                .collect()
        }
    }

    struct Fields(HashMap<String, String>);

    impl Visit for Fields {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0.insert(field.name().into(), format!("{value:?}"));
        }
        fn record_str(&mut self, field: &Field, value: &str) {
            self.0.insert(field.name().into(), value.into());
        }
    }

    struct CaptureLayer;

    impl<S: Subscriber + for<'a> LookupSpan<'a>> Layer<S> for CaptureLayer {
        fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
            let mut fields = Fields(HashMap::new());
            attrs.record(&mut fields);
            if let Some(span) = ctx.span(id) {
                span.extensions_mut().insert(fields);
            }
        }

        fn on_record(&self, id: &Id, values: &Record<'_>, ctx: Context<'_, S>) {
            if let Some(span) = ctx.span(id) {
                if let Some(fields) = span.extensions_mut().get_mut::<Fields>() {
                    values.record(fields);
                }
            }
        }

        fn on_close(&self, id: Id, ctx: Context<'_, S>) {
            let Some(span) = ctx.span(&id) else { return };
            let fields = span
                .extensions_mut()
                .remove::<Fields>()
                .map(|f| f.0)
                .unwrap_or_default();
            let record = SpanRecord {
                name: span.name(),
                fields,
                parent: span.parent().map(|p| p.name()),
            };
            SINK.with(|sink| {
                if let Some(captured) = sink.borrow().as_ref() {
                    captured.0.lock().unwrap().push(record);
                }
            });
        }
    }

    thread_local! {
        static SINK: std::cell::RefCell<Option<Captured>> = const { std::cell::RefCell::new(None) };
    }

    /// Stops capturing on this thread when dropped.
    pub struct CaptureGuard(());

    impl Drop for CaptureGuard {
        fn drop(&mut self) {
            SINK.with(|sink| sink.borrow_mut().take());
        }
    }

    /// Captures spans closed on the current thread until the guard is dropped. Use with
    /// `#[tokio::test]` (current-thread runtime) so every poll happens on this thread.
    ///
    /// A single global subscriber is installed once: per-test `set_default` subscribers
    /// race with tests running in parallel without one (tracing caches callsite interest
    /// globally), which made span assertions flaky.
    pub fn capture() -> (Captured, CaptureGuard) {
        static INSTALL: std::sync::Once = std::sync::Once::new();
        INSTALL.call_once(|| {
            let subscriber = tracing_subscriber::registry().with(CaptureLayer);
            tracing::subscriber::set_global_default(subscriber)
                .expect("no other global subscriber in tests");
        });
        let captured = Captured::default();
        SINK.with(|sink| *sink.borrow_mut() = Some(captured.clone()));
        (captured, CaptureGuard(()))
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::{testing::capture, *};
    use crate::{language_models::box_llm_events, schemas::ToolCall};

    fn result_with_usage() -> GenerateResult {
        GenerateResult {
            generation: "hi".into(),
            tokens: Some(TokenUsage::new(7, 3)),
            ..Default::default()
        }
        .with_tool_calls(vec![
            ToolCall::new("1", "a", "{}"),
            ToolCall::new("2", "b", "{}"),
        ])
    }

    #[tokio::test]
    async fn traced_generate_records_model_usage_and_tool_calls() {
        let (spans, _guard) = capture();
        let out = traced_generate("openai", "gpt-x", async { Ok(result_with_usage()) })
            .await
            .unwrap();
        assert_eq!(out.tool_calls.len(), 2);

        let span = &spans.named("llm.generate")[0];
        assert_eq!(span.field("provider"), Some("openai"));
        assert_eq!(span.field("model"), Some("gpt-x"));
        assert_eq!(span.field("stream"), Some("false"));
        assert_eq!(span.field("prompt_tokens"), Some("7"));
        assert_eq!(span.field("completion_tokens"), Some("3"));
        assert_eq!(span.field("total_tokens"), Some("10"));
        assert_eq!(span.field("tool_calls"), Some("2"));
        assert_eq!(span.field("error"), None);
        // No cache activity: the cache fields stay empty.
        assert_eq!(span.field("cache_read_tokens"), None);
        assert_eq!(span.field("cache_creation_tokens"), None);
    }

    #[tokio::test]
    async fn traced_generate_records_cache_tokens() {
        let (spans, _guard) = capture();
        traced_generate("anthropic", "claude-x", async {
            Ok(GenerateResult {
                tokens: Some(TokenUsage::new(100, 1).with_cache_tokens(80, 0)),
                ..Default::default()
            })
        })
        .await
        .unwrap();
        let span = &spans.named("llm.generate")[0];
        assert_eq!(span.field("cache_read_tokens"), Some("80"));
        assert_eq!(span.field("cache_creation_tokens"), Some("0"));
    }

    #[tokio::test]
    async fn traced_generate_without_usage_leaves_token_fields_empty() {
        let (spans, _guard) = capture();
        traced_generate("ollama", "m", async { Ok(GenerateResult::default()) })
            .await
            .unwrap();
        let span = &spans.named("llm.generate")[0];
        assert_eq!(span.field("prompt_tokens"), None);
        assert_eq!(span.field("tool_calls"), None);
    }

    #[tokio::test]
    async fn traced_generate_records_errors_and_returns_them() {
        let (spans, _guard) = capture();
        let err = traced_generate("openai", "m", async {
            Err(LLMError::ContentNotFound("boom".into()))
        })
        .await
        .unwrap_err();
        assert!(matches!(err, LLMError::ContentNotFound(_)));
        let span = &spans.named("llm.generate")[0];
        assert!(span.field("error").unwrap().contains("boom"), "{span:?}");
    }

    #[tokio::test]
    async fn traced_event_stream_polls_inside_the_span() {
        let (spans, _guard) = capture();
        let events = box_llm_events(async_stream::try_stream! {
            // e.g. the HTTP client's own spans, created while the stream is polled.
            tracing::info_span!("inner").in_scope(|| {});
            yield LLMEvent::Done(GenerateResult::default());
        });
        let _: Vec<_> = traced_event_stream("openai", "m", events).collect().await;
        assert_eq!(spans.named("inner")[0].parent, Some("llm.generate"));
    }

    #[tokio::test]
    async fn inner_work_runs_inside_the_span() {
        let (spans, _guard) = capture();
        traced_generate("openai", "m", async {
            tracing::info_span!("inner").in_scope(|| {});
            Ok(GenerateResult::default())
        })
        .await
        .unwrap();
        assert_eq!(spans.named("inner")[0].parent, Some("llm.generate"));
    }

    #[tokio::test]
    async fn traced_event_stream_records_the_done_result() {
        let (spans, _guard) = capture();
        let events = box_llm_events(futures::stream::iter([
            Ok(LLMEvent::Text("h".into())),
            Ok(LLMEvent::Done(result_with_usage())),
        ]));
        let collected: Vec<_> = traced_event_stream("anthropic", "claude", events)
            .collect()
            .await;
        assert_eq!(collected.len(), 2, "events pass through untouched");

        let span = &spans.named("llm.generate")[0];
        assert_eq!(span.field("provider"), Some("anthropic"));
        assert_eq!(span.field("stream"), Some("true"));
        assert_eq!(span.field("total_tokens"), Some("10"));
        assert_eq!(span.field("tool_calls"), Some("2"));
    }

    #[tokio::test]
    async fn traced_event_stream_records_stream_errors() {
        let (spans, _guard) = capture();
        let events = box_llm_events(futures::stream::iter([Err::<LLMEvent, _>(
            LLMError::ContentNotFound("cut".into()),
        )]));
        let collected: Vec<_> = traced_event_stream("ollama", "m", events).collect().await;
        assert!(collected[0].is_err());
        let span = &spans.named("llm.generate")[0];
        assert!(span.field("error").unwrap().contains("cut"));
    }

    #[tokio::test]
    async fn traced_event_stream_span_closes_when_dropped_early() {
        let (spans, _guard) = capture();
        let events = box_llm_events(futures::stream::iter([
            Ok(LLMEvent::Text("a".into())),
            Ok(LLMEvent::Done(result_with_usage())),
        ]));
        let mut stream = traced_event_stream("openai", "m", events);
        stream.next().await;
        assert!(spans.named("llm.generate").is_empty(), "still open");
        drop(stream);
        let span = &spans.named("llm.generate")[0];
        assert_eq!(span.field("total_tokens"), None, "Done never polled");
    }

    #[tokio::test]
    async fn embed_spans_record_provider_model_and_count() {
        let (spans, _guard) = capture();
        traced_embed_documents("openai", "emb", 3, async { Ok::<_, LLMError>(()) })
            .await
            .unwrap();
        traced_embed_query("ollama", "nomic", async {
            Err::<(), _>(LLMError::ContentNotFound("x".into()))
        })
        .await
        .unwrap_err();

        let docs = &spans.named("embed.documents")[0];
        assert_eq!(docs.field("provider"), Some("openai"));
        assert_eq!(docs.field("model"), Some("emb"));
        assert_eq!(docs.field("count"), Some("3"));
        let query = &spans.named("embed.query")[0];
        assert_eq!(query.field("model"), Some("nomic"));
        assert!(query.field("error").is_some());
    }
}
