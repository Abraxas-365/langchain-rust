//! Offline tests for the Claude (Anthropic Messages API) client against a local mock server,
//! including a full tools-agent loop.
use std::sync::{Arc, Mutex as StdMutex};

use futures::StreamExt;
use mockito::Matcher;
use serde_json::{json, Value};

use crate::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::Chain,
    language_models::{llm::LLM, options::CallOptions, LLMError},
    llm::{AnthropicError, Claude},
    prompt_args,
    schemas::{messages::ImageContent, Message, ToolCall},
    tools::Tool,
};

fn claude(server: &mockito::ServerGuard) -> Claude {
    Claude::new()
        .with_api_key("sk-ant-test")
        .with_model("claude-test")
        .with_base_url(server.url())
}

fn message_body(content: Value, stop_reason: &str) -> String {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
        "content": content, "stop_reason": stop_reason, "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 5}
    })
    .to_string()
}

fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect()
}

#[tokio::test]
async fn claude_generate_sends_headers_and_parses_text() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/messages")
        .match_header("x-api-key", "sk-ant-test")
        .match_header("anthropic-version", "2023-06-01")
        .match_body(Matcher::PartialJson(json!({
            "model": "claude-test",
            "system": "be brief",
            "messages": [{"role": "user", "content": "hi"}]
        })))
        .with_body(message_body(
            json!([{"type": "text", "text": "Hello"}, {"type": "text", "text": "!"}]),
            "end_turn",
        ))
        .create_async()
        .await;

    let res = claude(&server)
        .generate(&[
            Message::new_system_message("be brief"),
            Message::new_human_message("hi"),
        ])
        .await
        .unwrap();
    mock.assert_async().await;
    assert_eq!(res.generation, "Hello!");
    let tokens = res.tokens.unwrap();
    assert_eq!(
        (
            tokens.prompt_tokens,
            tokens.completion_tokens,
            tokens.total_tokens
        ),
        (10, 5, 15)
    );
}

#[tokio::test]
async fn claude_sends_images_as_image_blocks_before_the_text() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/messages")
        .match_body(Matcher::PartialJson(json!({
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo"}},
                {"type": "image", "source": {"type": "base64", "media_type": "image/webp", "data": "AAAA"}},
                {"type": "text", "text": "What is this?"}
            ]}]
        })))
        .with_body(message_body(
            json!([{"type": "text", "text": "A cat"}]),
            "end_turn",
        ))
        .create_async()
        .await;

    let message = Message::new_human_message("What is this?").with_images(vec![
        ImageContent::from("https://example.com/cat.png"),
        ImageContent::from("iVBORw0KGgo"),
        ImageContent::from("data:image/webp;base64,AAAA"),
    ]);
    let res = claude(&server).generate(&[message]).await.unwrap();
    mock.assert_async().await;
    assert_eq!(res.generation, "A cat");
}

#[tokio::test]
async fn claude_generate_returns_tool_calls() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/messages")
        .with_body(message_body(
            json!([
                {"type": "text", "text": "Checking"},
                {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Lima"}}
            ]),
            "tool_use",
        ))
        .create_async()
        .await;

    let res = claude(&server).invoke("weather?").await.unwrap();
    let calls: Value = serde_json::from_str(&res).unwrap();
    assert_eq!(
        calls,
        json!([{"id": "toolu_1", "type": "function",
                "function": {"name": "weather", "arguments": "{\"city\":\"Lima\"}"}}])
    );

    let res = claude(&server)
        .generate(&[Message::new_human_message("weather?")])
        .await
        .unwrap();
    assert_eq!(
        res.tool_calls,
        vec![ToolCall::new("toolu_1", "weather", r#"{"city":"Lima"}"#)]
    );
}

#[tokio::test]
async fn claude_http_errors_use_the_api_error_body() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/messages")
        .with_status(400)
        .with_body(
            json!({"type": "error",
                   "error": {"type": "invalid_request_error", "message": "messages: bad"}})
            .to_string(),
        )
        .create_async()
        .await;
    let err = claude(&server).invoke("hi").await.unwrap_err();
    assert!(
        matches!(&err, LLMError::AnthropicError(AnthropicError::InvalidRequestError(m)) if m == "messages: bad"),
        "{err:?}"
    );
}

fn tool_use_stream() -> String {
    sse(&[
        json!({"type": "message_start", "message": {"usage": {"input_tokens": 4, "output_tokens": 1}}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "On it"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "ping"}),
        json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_9", "name": "weather", "input": {}}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": ""}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}),
        json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"Cusco\"}"}}),
        json!({"type": "content_block_stop", "index": 1}),
        json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 20}}),
        json!({"type": "message_stop"}),
    ])
}

#[tokio::test]
async fn claude_streaming_generate_collects_tool_calls() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/messages")
        .match_body(Matcher::PartialJson(json!({"stream": true})))
        .with_header("content-type", "text/event-stream")
        .with_body(tool_use_stream())
        .create_async()
        .await;

    let seen = Arc::new(StdMutex::new(String::new()));
    let sink = seen.clone();
    let llm = claude(&server).with_options(CallOptions::new().with_streaming_func(
        move |chunk: String| {
            let sink = sink.clone();
            async move {
                sink.lock().unwrap().push_str(&chunk);
                Ok(())
            }
        },
    ));
    let res = llm
        .generate(&[Message::new_human_message("hi")])
        .await
        .unwrap();
    assert_eq!(*seen.lock().unwrap(), "On it");
    let tokens = res.tokens.unwrap();
    assert_eq!((tokens.prompt_tokens, tokens.completion_tokens), (4, 20));
    let calls: Value = serde_json::from_str(&res.generation).unwrap();
    assert_eq!(calls[0]["id"], "toolu_9");
    assert_eq!(calls[0]["function"]["arguments"], "{\"city\":\"Cusco\"}");
    assert_eq!(res.tool_calls.len(), 1);
    assert_eq!(res.tool_calls[0].id, "toolu_9");
    assert_eq!(
        res.tool_calls[0].arguments_value(),
        json!({"city": "Cusco"})
    );
}

#[tokio::test]
async fn claude_stream_yields_text_deltas_and_errors() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/messages")
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}),
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
        ]))
        .create_async()
        .await;

    let items: Vec<_> = claude(&server)
        .stream(&[Message::new_human_message("hi")])
        .await
        .unwrap()
        .collect()
        .await;
    let text: String = items
        .iter()
        .filter_map(|r| r.as_ref().ok())
        .map(|d| d.content.clone())
        .collect();
    assert_eq!(text, "Hello");
    assert!(matches!(
        items.last().unwrap(),
        Err(LLMError::AnthropicError(AnthropicError::OverloadedError(m))) if m == "busy"
    ));
}

/// Tool that returns a canned answer and records what it was called with.
#[derive(Clone)]
struct Weather(Arc<StdMutex<Vec<Value>>>);

#[async_trait::async_trait]
impl Tool for Weather {
    fn name(&self) -> String {
        "weather".into()
    }
    fn description(&self) -> String {
        "Current weather for a city".into()
    }
    fn parameters(&self) -> Value {
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]})
    }
    async fn parse_input(&self, input: &str) -> Value {
        serde_json::from_str(input).unwrap_or(Value::Null)
    }
    async fn run(&self, input: Value) -> Result<String, Box<dyn std::error::Error>> {
        self.0.lock().unwrap().push(input);
        Ok("sunny, 22C".into())
    }
}

#[tokio::test]
async fn tools_agent_runs_a_tool_loop_with_claude() {
    let mut server = mockito::Server::new_async().await;
    // 2nd turn: the request carries the tool_use and its tool_result. Registered first so the
    // more specific matcher wins over the catch-all below.
    let second = server
        .mock("POST", "/v1/messages")
        .match_body(Matcher::PartialJson(json!({
            "messages": [
                {"role": "user", "content": "weather in Lima?"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Lima"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "sunny, 22C"}
                ]}
            ]
        })))
        .with_body(message_body(
            json!([{"type": "text", "text": "It is sunny and 22C in Lima."}]),
            "end_turn",
        ))
        .expect(1)
        .create_async()
        .await;
    // 1st turn: tools are advertised in Anthropic format; Claude asks for the tool.
    let first = server
        .mock("POST", "/v1/messages")
        .match_body(Matcher::PartialJson(json!({
            "tools": [{"name": "weather", "description": "Current weather for a city",
                       "input_schema": {"type": "object", "required": ["city"]}}],
            "messages": [{"role": "user", "content": "weather in Lima?"}]
        })))
        .with_body(message_body(
            json!([{"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {"city": "Lima"}}]),
            "tool_use",
        ))
        .expect(1)
        .create_async()
        .await;

    let calls = Arc::new(StdMutex::new(Vec::new()));
    let tool: Arc<dyn Tool> = Arc::new(Weather(calls.clone()));
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[tool])
        .build(claude(&server))
        .unwrap();
    let out = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "weather in Lima?"})
        .await
        .unwrap();

    assert_eq!(out, "It is sunny and 22C in Lima.");
    assert_eq!(*calls.lock().unwrap(), vec![json!({"city": "Lima"})]);
    first.assert_async().await;
    second.assert_async().await;
}

#[tokio::test]
async fn claude_generate_stream_yields_text_and_tool_calls() {
    use crate::language_models::LLMEvent;

    let mut server = mockito::Server::new_async().await;
    let stream = server
        .mock("POST", "/v1/messages")
        .match_body(Matcher::PartialJson(json!({"stream": true})))
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[
            json!({"type": "message_start", "message": {"usage": {"input_tokens": 7, "output_tokens": 1}}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Let me "}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "check."}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "weather", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"city\":"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"Lima\"}"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
            json!({"type": "message_stop"}),
        ]))
        .create_async()
        .await;

    let llm = claude(&server);
    let messages = [Message::new_human_message("weather?")];
    let events: Vec<_> = llm
        .generate_stream(&messages)
        .map(Result::unwrap)
        .collect()
        .await;
    assert!(matches!(&events[0], LLMEvent::Text(t) if t == "Let me "));
    assert!(matches!(&events[1], LLMEvent::Text(t) if t == "check."));
    let LLMEvent::Done(result) = &events[2] else {
        panic!("{events:?}")
    };
    assert_eq!(events.len(), 3);
    assert_eq!(result.tool_calls.len(), 1);
    assert_eq!(result.tool_calls[0].id, "toolu_1");
    assert_eq!(
        result.tool_calls[0].arguments_value(),
        json!({"city": "Lima"})
    );
    let tokens = result.tokens.as_ref().unwrap();
    assert_eq!((tokens.prompt_tokens, tokens.completion_tokens), (7, 9));
    stream.assert_async().await;
}

#[tokio::test]
async fn claude_generate_stream_ends_with_error_event() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/messages")
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}}),
            json!({"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}),
        ]))
        .create_async()
        .await;

    let llm = claude(&server);
    let messages = [Message::new_human_message("hi")];
    let events: Vec<_> = llm.generate_stream(&messages).collect().await;
    assert_eq!(events.len(), 2, "{events:?}");
    assert!(events[0].is_ok());
    assert!(matches!(
        &events[1],
        Err(LLMError::AnthropicError(AnthropicError::OverloadedError(m))) if m == "busy"
    ));
}

// ---- tracing ----

#[tokio::test]
async fn claude_generate_and_generate_stream_run_in_llm_spans() {
    use crate::observability::testing::capture;

    let mut server = mockito::Server::new_async().await;
    // More specific mock first: streaming requests carry "stream": true.
    server
        .mock("POST", "/v1/messages")
        .match_body(Matcher::PartialJson(json!({"stream": true})))
        .with_header("content-type", "text/event-stream")
        .with_body(tool_use_stream())
        .create_async()
        .await;
    server
        .mock("POST", "/v1/messages")
        .with_body(message_body(
            json!([{"type": "text", "text": "Hi"}]),
            "end_turn",
        ))
        .create_async()
        .await;

    let (spans, _guard) = capture();
    let llm = claude(&server);
    let messages = [Message::new_human_message("hi")];
    llm.generate(&messages).await.unwrap();
    let events: Vec<_> = llm.generate_stream(&messages).collect().await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");

    let llm_spans = spans.named("llm.generate");
    assert_eq!(llm_spans.len(), 2);
    for span in &llm_spans {
        assert_eq!(span.field("provider"), Some("anthropic"));
        assert_eq!(span.field("model"), Some("claude-test"));
    }
    assert_eq!(llm_spans[0].field("stream"), Some("false"));
    assert_eq!(llm_spans[0].field("total_tokens"), Some("15"));
    assert_eq!(llm_spans[1].field("stream"), Some("true"));
    assert_eq!(llm_spans[1].field("tool_calls"), Some("1"));
}
