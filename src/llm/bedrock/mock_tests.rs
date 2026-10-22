//! Offline tests: a real Bedrock SDK client pointed at a mockito server, covering the
//! Converse JSON API and the binary event stream used by ConverseStream.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use aws_sdk_bedrockruntime::config::{retry::RetryConfig, Credentials, Region};
use aws_smithy_types::event_stream::{Header, HeaderValue, Message as EventMessage};
use futures::StreamExt;
use mockito::Matcher;
use serde_json::{json, Value};

use super::*;
use crate::tools::Tool;

const CONVERSE: &str = "/model/test-model/converse";
const CONVERSE_STREAM: &str = "/model/test-model/converse-stream";

fn bedrock_for(server: &mockito::ServerGuard) -> Bedrock {
    let conf = aws_sdk_bedrockruntime::Config::builder()
        .behavior_version_latest()
        .region(Region::new("us-east-1"))
        .endpoint_url(server.url())
        .credentials_provider(Credentials::new("AKID", "SECRET", None, None, "test"))
        .retry_config(RetryConfig::disabled())
        .build();
    Bedrock::default()
        .with_model(BedrockModel::Custom("test-model".into()))
        .with_client(BedrockClient::from_conf(conf))
}

fn converse_response(content: Value) -> String {
    json!({
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
        "metrics": {"latencyMs": 1}
    })
    .to_string()
}

/// Encodes ConverseStream events as an AWS event stream body.
fn event_stream(events: &[(&'static str, Value)]) -> Vec<u8> {
    let mut body = Vec::new();
    for (event_type, payload) in events {
        let message = EventMessage::new(payload.to_string().into_bytes())
            .add_header(Header::new(
                ":message-type",
                HeaderValue::String("event".into()),
            ))
            .add_header(Header::new(
                ":event-type",
                HeaderValue::String((*event_type).into()),
            ))
            .add_header(Header::new(
                ":content-type",
                HeaderValue::String("application/json".into()),
            ));
        aws_smithy_eventstream::frame::write_message_to(&message, &mut body).unwrap();
    }
    body
}

fn weather_tool() -> FunctionDefinition {
    FunctionDefinition::new(
        "weather",
        "Current weather for a city",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    )
}

#[tokio::test]
async fn generate_sends_converse_request_and_reads_text() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", CONVERSE)
        .match_body(Matcher::PartialJson(json!({
            "system": [{"text": "be brief"}],
            "messages": [{"role": "user", "content": [{"text": "hi"}]}],
            "inferenceConfig": {"maxTokens": 64}
        })))
        .with_header("content-type", "application/json")
        .with_body(converse_response(json!([{"text": "Hel"}, {"text": "lo"}])))
        .create_async()
        .await;

    let llm = bedrock_for(&server).with_max_tokens(64);
    let out = llm
        .generate(&[
            Message::new_system_message("be brief"),
            Message::new_human_message("hi"),
        ])
        .await
        .unwrap();

    assert_eq!(out.generation, "Hello");
    assert_eq!(out.tokens.unwrap().total_tokens, 5);
    mock.assert_async().await;
}

#[tokio::test]
async fn generate_sends_tools_and_returns_tool_calls() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", CONVERSE)
        .match_body(Matcher::PartialJson(json!({
            "toolConfig": {
                "tools": [{"toolSpec": {
                    "name": "weather",
                    "description": "Current weather for a city",
                    "inputSchema": {"json": {"type": "object", "required": ["city"]}}
                }}],
                "toolChoice": {"tool": {"name": "weather"}}
            }
        })))
        .with_header("content-type", "application/json")
        .with_body(converse_response(json!([
            {"text": "Let me check."},
            {"toolUse": {"toolUseId": "tu_1", "name": "weather", "input": {"city": "Lima"}}}
        ])))
        .create_async()
        .await;

    let mut llm = bedrock_for(&server).with_tools(vec![weather_tool()]);
    llm.add_options(
        CallOptions::new()
            .with_function_call_behavior(FunctionCallBehavior::Named("weather".into())),
    );
    let out = llm.invoke("weather in Lima?").await.unwrap();

    let calls: Vec<FunctionCallResponse> = serde_json::from_str(&out).unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "tu_1");
    assert_eq!(calls[0].function.name, "weather");
    let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args, json!({"city": "Lima"}));
    mock.assert_async().await;
}

#[tokio::test]
async fn api_errors_surface_the_service_message() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", CONVERSE)
        .with_status(400)
        .with_header("content-type", "application/json")
        .with_header("x-amzn-errortype", "ValidationException")
        .with_body(r#"{"message":"The model ID is invalid"}"#)
        .create_async()
        .await;

    let err = bedrock_for(&server).invoke("hi").await.unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("ValidationException"), "{msg}");
    assert!(msg.contains("The model ID is invalid"), "{msg}");
}

fn text_stream_body() -> Vec<u8> {
    event_stream(&[
        ("messageStart", json!({"role": "assistant"})),
        (
            "contentBlockDelta",
            json!({"contentBlockIndex": 0, "delta": {"text": "Hel"}}),
        ),
        (
            "contentBlockDelta",
            json!({"contentBlockIndex": 0, "delta": {"text": "lo"}}),
        ),
        ("contentBlockStop", json!({"contentBlockIndex": 0})),
        ("messageStop", json!({"stopReason": "end_turn"})),
        (
            "metadata",
            json!({"usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
                   "metrics": {"latencyMs": 1}}),
        ),
    ])
}

#[tokio::test]
async fn stream_yields_text_deltas_and_usage() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", CONVERSE_STREAM)
        .with_header("content-type", "application/vnd.amazon.eventstream")
        .with_body(text_stream_body())
        .create_async()
        .await;

    let mut stream = bedrock_for(&server)
        .stream(&[Message::new_human_message("hi")])
        .await
        .unwrap();
    let mut text = String::new();
    let mut usage = None;
    while let Some(item) = stream.next().await {
        let item = item.unwrap();
        text.push_str(&item.content);
        if item.tokens.is_some() {
            usage = item.tokens;
        }
    }
    assert_eq!(text, "Hello");
    assert_eq!(usage.unwrap().total_tokens, 5);
}

#[tokio::test]
async fn generate_with_streaming_func_rebuilds_tool_calls() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", CONVERSE_STREAM)
        .with_header("content-type", "application/vnd.amazon.eventstream")
        .with_body(event_stream(&[
            ("messageStart", json!({"role": "assistant"})),
            (
                "contentBlockDelta",
                json!({"contentBlockIndex": 0, "delta": {"text": "Checking."}}),
            ),
            ("contentBlockStop", json!({"contentBlockIndex": 0})),
            (
                "contentBlockStart",
                json!({"contentBlockIndex": 1,
                       "start": {"toolUse": {"toolUseId": "tu_9", "name": "weather"}}}),
            ),
            (
                "contentBlockDelta",
                json!({"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{\"city\":"}}}),
            ),
            (
                "contentBlockDelta",
                json!({"contentBlockIndex": 1, "delta": {"toolUse": {"input": "\"Lima\"}"}}}),
            ),
            ("contentBlockStop", json!({"contentBlockIndex": 1})),
            ("messageStop", json!({"stopReason": "tool_use"})),
            (
                "metadata",
                json!({"usage": {"inputTokens": 4, "outputTokens": 6, "totalTokens": 10},
                       "metrics": {"latencyMs": 1}}),
            ),
        ]))
        .create_async()
        .await;

    let seen = Arc::new(Mutex::new(String::new()));
    let sink = seen.clone();
    let llm = bedrock_for(&server).with_options(CallOptions::new().with_streaming_func(
        move |chunk: String| {
            sink.lock().unwrap().push_str(&chunk);
            async { Ok(()) }
        },
    ));
    let out = llm
        .generate(&[Message::new_human_message("weather?")])
        .await
        .unwrap();

    assert_eq!(*seen.lock().unwrap(), "Checking.");
    assert_eq!(out.tokens.unwrap().total_tokens, 10);
    let calls: Vec<FunctionCallResponse> = serde_json::from_str(&out.generation).unwrap();
    assert_eq!(calls[0].id, "tu_9");
    assert_eq!(calls[0].function.arguments, "{\"city\":\"Lima\"}");
}

#[derive(Clone)]
struct Weather(Arc<Mutex<Vec<Value>>>);

#[async_trait]
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
        serde_json::from_str(input).unwrap_or_default()
    }
    async fn run(&self, input: Value) -> Result<String, Box<dyn std::error::Error>> {
        self.0.lock().unwrap().push(input);
        Ok("sunny".into())
    }
}

#[tokio::test]
async fn tools_agent_runs_a_tool_loop_with_bedrock() {
    use crate::{
        agent::{AgentExecutor, OpenAiToolAgentBuilder},
        chain::Chain,
        prompt_args,
    };

    let mut server = mockito::Server::new_async().await;
    // 2nd turn (registered first so it wins once its body matches): the tool call and
    // its result go back as toolUse / toolResult blocks.
    let second = server
        .mock("POST", CONVERSE)
        .match_body(Matcher::PartialJson(json!({
            "messages": [
                {"role": "user", "content": [{"text": "weather in Lima?"}]},
                {"role": "assistant", "content": [
                    {"toolUse": {"toolUseId": "tu_1", "name": "weather", "input": {"city": "Lima"}}}
                ]},
                {"role": "user", "content": [
                    {"toolResult": {"toolUseId": "tu_1", "content": [{"text": "sunny"}]}}
                ]}
            ]
        })))
        .with_header("content-type", "application/json")
        .with_body(converse_response(json!([{"text": "Sunny in Lima."}])))
        .expect(1)
        .create_async()
        .await;
    let first = server
        .mock("POST", CONVERSE)
        .match_body(Matcher::PartialJson(json!({
            "toolConfig": {"tools": [{"toolSpec": {"name": "weather"}}]},
            "messages": [{"role": "user", "content": [{"text": "weather in Lima?"}]}]
        })))
        .with_header("content-type", "application/json")
        .with_body(converse_response(json!([
            {"toolUse": {"toolUseId": "tu_1", "name": "weather", "input": {"city": "Lima"}}}
        ])))
        .expect(1)
        .create_async()
        .await;

    let calls = Arc::new(Mutex::new(Vec::new()));
    let tool: Arc<dyn Tool> = Arc::new(Weather(calls.clone()));
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[tool])
        .build(bedrock_for(&server))
        .unwrap();
    let out = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "weather in Lima?"})
        .await
        .unwrap();

    assert_eq!(out, "Sunny in Lima.");
    assert_eq!(*calls.lock().unwrap(), vec![json!({"city": "Lima"})]);
    first.assert_async().await;
    second.assert_async().await;
}
