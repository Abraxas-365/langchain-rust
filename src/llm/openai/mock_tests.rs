//! Offline tests for the OpenAI client: assert on the JSON wire format and on responses
//! from a local mock server, so they stay valid across async-openai upgrades.
use super::*;
use crate::schemas::{FunctionCallBehavior, FunctionDefinition, ResponseFormat};
use serde_json::{json, Value};

fn request_json(llm: &OpenAI<OpenAIConfig>, messages: &[Message], stream: bool) -> Value {
    serde_json::to_value(llm.generate_request(messages, stream).unwrap()).unwrap()
}

fn llm_for(server: &mockito::ServerGuard) -> OpenAI<OpenAIConfig> {
    OpenAI::new(
        OpenAIConfig::new()
            .with_api_base(server.url())
            .with_api_key("test-key"),
    )
}

#[test]
fn model_names() {
    assert_eq!(OpenAIModel::Gpt35.to_string(), "gpt-3.5-turbo");
    assert_eq!(OpenAIModel::Gpt4.to_string(), "gpt-4");
    assert_eq!(OpenAIModel::Gpt4Turbo.to_string(), "gpt-4-turbo-preview");
    assert_eq!(OpenAIModel::Gpt4o.to_string(), "gpt-4o");
    let s: String = OpenAIModel::Gpt4oMini.into();
    assert_eq!(s, "gpt-4o-mini");
}

#[test]
fn request_maps_all_message_types() {
    let tool_calls = json!([{
        "id": "call_1",
        "type": "function",
        "function": {"name": "calc", "arguments": "{\"x\":1}"}
    }]);
    let messages = vec![
        Message::new_system_message("be nice"),
        Message::new_human_message("hi"),
        Message::new_ai_message("").with_tool_calls(tool_calls.clone()),
        Message::new_tool_message("2", "call_1"),
        Message::new_ai_message("done"),
    ];
    let req = request_json(&OpenAI::default(), &messages, false);

    assert_eq!(req["model"], "gpt-4o-mini");
    let msgs = req["messages"].as_array().unwrap();
    assert_eq!(msgs.len(), 5);
    assert_eq!(msgs[0], json!({"role": "system", "content": "be nice"}));
    assert_eq!(msgs[1], json!({"role": "user", "content": "hi"}));
    assert_eq!(msgs[2]["role"], "assistant");
    assert_eq!(msgs[2]["tool_calls"], tool_calls);
    assert_eq!(
        msgs[3],
        json!({"role": "tool", "content": "2", "tool_call_id": "call_1"})
    );
    assert_eq!(msgs[4]["content"], "done");
}

#[test]
fn request_maps_images_to_content_parts() {
    let messages = vec![Message::new_human_message_with_images(vec![
        "data:image/png;base64,AAA",
    ])];
    let req = request_json(&OpenAI::default(), &messages, false);
    // No text: only the image part, no empty text part.
    assert_eq!(
        req["messages"][0]["content"],
        json!([{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}])
    );
}

#[test]
fn request_sends_text_and_images_together() {
    let messages = vec![
        Message::new_human_message("What is this?").with_images(vec![
            ImageContent::from("https://example.com/cat.png").with_detail("LOW"),
            // Bare base64 gets a data: URL with the sniffed media type.
            ImageContent::from("/9j/4AAQ"),
        ]),
    ];
    let req = request_json(&OpenAI::default(), &messages, false);
    assert_eq!(
        req["messages"][0]["content"],
        json!([
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png", "detail": "low"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ"}}
        ])
    );

    // `Some(vec![])` is plain text, not an empty parts list.
    let mut empty = Message::new_human_message("hi");
    empty.images = Some(vec![]);
    let req = request_json(&OpenAI::default(), &[empty], false);
    assert_eq!(req["messages"][0]["content"], "hi");

    for (detail, expected) in [("auto", "auto"), ("high", "high")] {
        let m = Message::new_human_message("x")
            .with_images(vec![
                ImageContent::from("https://a/b.png").with_detail(detail)
            ]);
        let req = request_json(&OpenAI::default(), &[m], false);
        assert_eq!(
            req["messages"][0]["content"][1]["image_url"]["detail"],
            expected
        );
    }

    let bad = Message::new_human_message("x")
        .with_images(vec![
            ImageContent::from("https://a/b.png").with_detail("ultra")
        ]);
    let err = OpenAI::default()
        .generate_request(&[bad], false)
        .unwrap_err();
    assert!(err.to_string().contains("ultra"), "{err}");
}

#[test]
fn request_includes_options() {
    let options = CallOptions::new()
        .with_temperature(0.5)
        .with_max_tokens(64)
        .with_stop_words(vec!["STOP".into()])
        .with_stream_usage(true)
        .with_functions(vec![FunctionDefinition::new(
            "my tool",
            "does things",
            json!({"type": "object", "properties": {}}),
        )])
        .with_function_call_behavior(FunctionCallBehavior::Named("my_tool".into()))
        .with_response_format(ResponseFormat::JsonSchema {
            description: None,
            name: "out".into(),
            schema: Some(json!({"type": "object"})),
            strict: Some(true),
        });
    let llm = OpenAI::default()
        .with_model("custom-model")
        .with_options(options);

    let req = request_json(&llm, &[Message::new_human_message("x")], true);
    assert_eq!(req["model"], "custom-model");
    assert_eq!(req["temperature"], 0.5);
    assert_eq!(req["max_tokens"], 64);
    assert_eq!(req["stop"], json!(["STOP"]));
    assert_eq!(req["stream_options"]["include_usage"], true);
    assert_eq!(req["tools"][0]["type"], "function");
    assert_eq!(req["tools"][0]["function"]["name"], "my_tool");
    assert_eq!(req["tools"][0]["function"]["description"], "does things");
    assert_eq!(
        req["tool_choice"],
        json!({"type": "function", "function": {"name": "my_tool"}})
    );
    assert_eq!(req["response_format"]["type"], "json_schema");
    assert_eq!(req["response_format"]["json_schema"]["name"], "out");
    assert_eq!(req["response_format"]["json_schema"]["strict"], true);

    // stream_options is only sent for streaming requests
    let req = request_json(&llm, &[Message::new_human_message("x")], false);
    assert!(req.get("stream_options").is_none_or(Value::is_null));
}

#[test]
fn thinking_effort_maps_to_reasoning_effort() {
    use crate::language_models::{ThinkingConfig, ThinkingEffort};
    let effort_for = |config: Option<ThinkingConfig>| {
        let mut options = CallOptions::new();
        if let Some(config) = config {
            options = options.with_thinking(config);
        }
        let llm = OpenAI::default()
            .with_model("o4-mini")
            .with_options(options);
        let req = request_json(&llm, &[Message::new_human_message("x")], false);
        req.get("reasoning_effort").cloned().unwrap_or(Value::Null)
    };
    for (effort, wire) in [
        (ThinkingEffort::Low, "low"),
        (ThinkingEffort::Medium, "medium"),
        (ThinkingEffort::High, "high"),
        (ThinkingEffort::XHigh, "xhigh"),
        (ThinkingEffort::Max, "max"),
    ] {
        assert_eq!(
            effort_for(Some(ThinkingConfig::enabled().with_effort(effort))),
            wire
        );
    }
    // No effort, thinking off, or no config: the parameter is left out.
    assert_eq!(effort_for(Some(ThinkingConfig::enabled())), Value::Null);
    assert_eq!(
        effort_for(Some(
            ThinkingConfig::disabled().with_effort(ThinkingEffort::High)
        )),
        Value::Null
    );
    assert_eq!(effort_for(None), Value::Null);
}

#[test]
fn tool_choice_auto_none_and_simple_formats() {
    for (behavior, expected) in [
        (FunctionCallBehavior::Auto, json!("auto")),
        (FunctionCallBehavior::None, json!("none")),
    ] {
        let llm = OpenAI::default()
            .with_options(CallOptions::new().with_function_call_behavior(behavior));
        let req = request_json(&llm, &[Message::new_human_message("x")], false);
        assert_eq!(req["tool_choice"], expected);
    }
    for (format, expected) in [
        (ResponseFormat::Text, "text"),
        (ResponseFormat::JsonObject, "json_object"),
    ] {
        let llm = OpenAI::default().with_options(CallOptions::new().with_response_format(format));
        let req = request_json(&llm, &[Message::new_human_message("x")], false);
        assert_eq!(req["response_format"]["type"], expected);
    }
}

#[test]
fn tool_choice_named_function() {
    let llm = OpenAI::default().with_options(
        CallOptions::new()
            .with_function_call_behavior(FunctionCallBehavior::Named("my_tool".into())),
    );
    let req = request_json(&llm, &[Message::new_human_message("x")], false);
    assert_eq!(
        req["tool_choice"],
        json!({"type": "function", "function": {"name": "my_tool"}})
    );
}

#[test]
fn json_schema_without_schema_sends_empty_object() {
    let llm = OpenAI::default().with_options(CallOptions::new().with_response_format(
        ResponseFormat::JsonSchema {
            description: None,
            name: "out".into(),
            schema: None,
            strict: None,
        },
    ));
    let req = request_json(&llm, &[Message::new_human_message("x")], false);
    assert_eq!(
        req["response_format"],
        json!({"type": "json_schema", "json_schema": {"name": "out", "schema": {}}})
    );
}

#[test]
fn invalid_tool_calls_value_is_an_error() {
    let messages = vec![Message::new_ai_message("").with_tool_calls(json!("not a list"))];
    assert!(OpenAI::default()
        .generate_request(&messages, false)
        .is_err());
}

fn completion_body(message: Value) -> String {
    json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop", "logprobs": null}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
    })
    .to_string()
}

#[tokio::test]
async fn generate_against_mock_server() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/chat/completions")
        .match_header("authorization", "Bearer test-key")
        .match_body(mockito::Matcher::PartialJson(json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hello"}]
        })))
        .with_header("content-type", "application/json")
        .with_body(completion_body(
            json!({"role": "assistant", "content": "Hi there"}),
        ))
        .create_async()
        .await;

    let result = llm_for(&server)
        .generate(&[Message::new_human_message("hello")])
        .await
        .unwrap();
    mock.assert_async().await;
    assert_eq!(result.generation, "Hi there");
    let tokens = result.tokens.unwrap();
    assert_eq!(
        (
            tokens.prompt_tokens,
            tokens.completion_tokens,
            tokens.total_tokens
        ),
        (5, 7, 12)
    );
}

#[tokio::test]
async fn generate_returns_tool_calls_as_json() {
    let mut server = mockito::Server::new_async().await;
    let tool_calls = json!([{
        "id": "call_9",
        "type": "function",
        "function": {"name": "calc", "arguments": "{\"x\":2}"}
    }]);
    server
        .mock("POST", "/chat/completions")
        .with_header("content-type", "application/json")
        .with_body(completion_body(
            json!({"role": "assistant", "content": null, "tool_calls": tool_calls}),
        ))
        .create_async()
        .await;

    let result = llm_for(&server).invoke("compute").await.unwrap();
    let parsed: Value = serde_json::from_str(&result).unwrap();
    assert_eq!(parsed, tool_calls);

    let res = llm_for(&server)
        .generate(&[Message::new_human_message("compute")])
        .await
        .unwrap();
    assert_eq!(
        res.tool_calls,
        vec![crate::schemas::ToolCall::new(
            "call_9",
            "calc",
            r#"{"x":2}"#
        )]
    );
}

#[tokio::test]
async fn generate_surfaces_api_errors() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat/completions")
        .with_status(401)
        .with_header("content-type", "application/json")
        .with_body(
            json!({"error": {"message": "bad key", "type": "invalid_request_error", "param": null, "code": "invalid_api_key"}})
                .to_string(),
        )
        .create_async()
        .await;

    let err = llm_for(&server).invoke("x").await.unwrap_err();
    assert!(matches!(err, LLMError::OpenAIError(_)), "{err:?}");
    assert!(err.to_string().contains("bad key"), "{err}");
}

fn sse(chunks: &[Value]) -> String {
    let mut body: String = chunks.iter().map(|c| format!("data: {}\n\n", c)).collect();
    body.push_str("data: [DONE]\n\n");
    body
}

fn chunk(content: Option<&str>, usage: Option<Value>) -> Value {
    let choices = match content {
        Some(c) => {
            json!([{"index": 0, "delta": {"role": "assistant", "content": c}, "finish_reason": null}])
        }
        None => json!([]),
    };
    json!({
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1,
        "model": "gpt-4o-mini", "choices": choices, "usage": usage
    })
}

#[tokio::test]
async fn stream_against_mock_server() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({"stream": true})))
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[
            chunk(Some("Hel"), None),
            chunk(Some("lo"), None),
            chunk(
                None,
                Some(json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})),
            ),
        ]))
        .create_async()
        .await;

    let llm = llm_for(&server).with_options(CallOptions::new().with_stream_usage(true));
    let items: Vec<_> = llm
        .stream(&[Message::new_human_message("hi")])
        .await
        .unwrap()
        .collect()
        .await;
    let items: Vec<StreamData> = items.into_iter().map(Result::unwrap).collect();
    let text: String = items.iter().map(|d| d.content.as_str()).collect();
    assert_eq!(text, "Hello");
    let usage = items.last().unwrap().tokens.clone().unwrap();
    assert_eq!(usage.total_tokens, 3);
}

// #290: some OpenAI-compatible servers attach usage to every chunk,
// the content of those chunks must not be dropped.
#[tokio::test]
async fn stream_keeps_content_of_chunks_that_carry_usage() {
    let mut server = mockito::Server::new_async().await;
    let usage = json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3});
    server
        .mock("POST", "/chat/completions")
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[
            chunk(Some("Hel"), Some(usage.clone())),
            chunk(Some("lo"), Some(usage)),
        ]))
        .create_async()
        .await;

    let items: Vec<StreamData> = llm_for(&server)
        .stream(&[Message::new_human_message("hi")])
        .await
        .unwrap()
        .map(Result::unwrap)
        .collect()
        .await;
    let text: String = items.iter().map(|d| d.content.as_str()).collect();
    assert_eq!(text, "Hello");
    assert_eq!(items[1].tokens.as_ref().unwrap().total_tokens, 3);
}

#[tokio::test]
async fn generate_with_streaming_func_collects_chunks() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({"stream": true})))
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[chunk(Some("a"), None), chunk(Some("b"), None)]))
        .create_async()
        .await;

    let seen = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::<String>::new()));
    let seen_c = seen.clone();
    let llm =
        llm_for(&server).with_options(CallOptions::new().with_streaming_func(move |s: String| {
            let seen = seen_c.clone();
            async move {
                seen.lock().await.push(s);
                Ok(())
            }
        }));
    let result = llm.invoke("hi").await.unwrap();
    assert_eq!(result, "ab");
    assert_eq!(seen.lock().await.len(), 2);
}

// ---- generate_stream ----

use crate::language_models::LLMEvent;

fn tool_delta(index: u32, id: Option<&str>, name: Option<&str>, args: &str) -> Value {
    let mut call = json!({"index": index, "function": {"arguments": args}});
    if let Some(id) = id {
        call["id"] = json!(id);
        call["type"] = json!("function");
    }
    if let Some(name) = name {
        call["function"]["name"] = json!(name);
    }
    json!({
        "id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": null}]
    })
}

async fn stream_events_from(server: &mockito::ServerGuard) -> Vec<Result<LLMEvent, LLMError>> {
    let llm = llm_for(server);
    let messages = [Message::new_human_message("hi")];
    llm.generate_stream(&messages).collect().await
}

#[tokio::test]
async fn generate_stream_yields_text_then_done_with_usage() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({"stream": true})))
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[
            chunk(Some("Hel"), None),
            chunk(Some(""), None),
            chunk(Some("lo"), None),
            chunk(
                None,
                Some(json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})),
            ),
        ]))
        .create_async()
        .await;

    let events: Vec<_> = stream_events_from(&server)
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(events.len(), 3, "{events:?}");
    assert!(matches!(&events[0], LLMEvent::Text(t) if t == "Hel"));
    assert!(matches!(&events[1], LLMEvent::Text(t) if t == "lo"));
    let LLMEvent::Done(result) = &events[2] else {
        panic!("{events:?}")
    };
    assert_eq!(result.generation, "Hello");
    assert!(result.tool_calls.is_empty());
    assert_eq!(result.tokens.as_ref().unwrap().total_tokens, 3);
}

#[tokio::test]
async fn generate_stream_rebuilds_tool_calls_from_deltas() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat/completions")
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[
            tool_delta(0, Some("call_a"), Some("weather"), ""),
            tool_delta(1, Some("call_b"), Some("time"), "{\"tz\":"),
            tool_delta(0, None, None, "{\"city\":"),
            tool_delta(0, None, None, "\"Lima\"}"),
            tool_delta(1, None, None, "\"UTC\"}"),
        ]))
        .create_async()
        .await;

    let events: Vec<_> = stream_events_from(&server)
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    // No text at all: tool-call JSON is never streamed as answer text.
    assert_eq!(events.len(), 1, "{events:?}");
    let LLMEvent::Done(result) = &events[0] else {
        panic!("{events:?}")
    };
    let calls: Vec<_> = result
        .tool_calls
        .iter()
        .map(|c| (c.id.as_str(), c.name.as_str(), c.arguments.as_str()))
        .collect();
    assert_eq!(
        calls,
        vec![
            ("call_a", "weather", "{\"city\":\"Lima\"}"),
            ("call_b", "time", "{\"tz\":\"UTC\"}"),
        ]
    );
}

#[tokio::test]
async fn generate_stream_yields_http_errors_as_items() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat/completions")
        .with_status(401)
        .with_header("content-type", "application/json")
        .with_body(r#"{"error":{"message":"bad key","type":"invalid_request_error","param":null,"code":null}}"#)
        .create_async()
        .await;

    let events = stream_events_from(&server).await;
    assert_eq!(events.len(), 1, "{events:?}");
    assert!(events[0].is_err(), "{events:?}");
}

#[tokio::test]
async fn agent_stream_events_with_openai_streams_answer_tokens() {
    use crate::{
        agent::{AgentExecutor, AgentStreamEvent, OpenAiToolAgentBuilder},
        prompt_args,
    };

    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat/completions")
        .match_body(mockito::Matcher::PartialJson(json!({"stream": true})))
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[chunk(Some("Hi "), None), chunk(Some("there"), None)]))
        .expect(1)
        .create_async()
        .await;

    let agent = OpenAiToolAgentBuilder::new()
        .build(llm_for(&server))
        .unwrap();
    let executor = AgentExecutor::from_agent(agent);
    let events: Vec<_> = executor
        .stream_events(prompt_args! {"input" => "hi"})
        .map(Result::unwrap)
        .collect()
        .await;
    assert_eq!(
        events,
        vec![
            AgentStreamEvent::Text("Hi ".into()),
            AgentStreamEvent::Text("there".into()),
            AgentStreamEvent::Finish {
                output: "Hi there".into()
            },
        ]
    );
}

// ---- tracing ----

use crate::observability::testing::capture;

#[tokio::test]
async fn generate_runs_in_an_llm_span_with_usage() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat/completions")
        .with_header("content-type", "application/json")
        .with_body(completion_body(
            json!({"role": "assistant", "content": "ok"}),
        ))
        .create_async()
        .await;

    let (spans, _guard) = capture();
    llm_for(&server).invoke("hi").await.unwrap();

    let span = &spans.named("llm.generate")[0];
    assert_eq!(span.field("provider"), Some("openai"));
    assert_eq!(span.field("model"), Some("gpt-4o-mini"));
    assert_eq!(span.field("stream"), Some("false"));
    assert_eq!(span.field("total_tokens"), Some("12"));
}

#[tokio::test]
async fn generate_stream_runs_in_an_llm_span_with_usage() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat/completions")
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&[
            chunk(Some("a"), None),
            chunk(
                None,
                Some(json!({"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})),
            ),
        ]))
        .create_async()
        .await;

    let (spans, _guard) = capture();
    let events = stream_events_from(&server).await;
    assert!(events.iter().all(Result::is_ok));

    let span = &spans.named("llm.generate")[0];
    assert_eq!(span.field("provider"), Some("openai"));
    assert_eq!(span.field("stream"), Some("true"));
    assert_eq!(span.field("total_tokens"), Some("3"));
}

#[tokio::test]
async fn generate_with_streaming_func_returns_stream_errors() {
    // A broken chunk used to be printed to stderr and skipped, returning a truncated answer.
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat/completions")
        .with_header("content-type", "text/event-stream")
        .with_body(format!(
            "data: {}\n\ndata: {{not json\n\ndata: [DONE]\n\n",
            chunk(Some("a"), None)
        ))
        .create_async()
        .await;

    let (spans, _guard) = capture();
    let llm = llm_for(&server)
        .with_options(CallOptions::new().with_streaming_func(|_: String| async { Ok(()) }));
    let err = llm.invoke("hi").await.unwrap_err();
    assert!(matches!(err, LLMError::OpenAIError(_)), "{err:?}");
    assert!(spans.named("llm.generate")[0].field("error").is_some());
}
