//! Offline tests for the hand-written HTTP LLM clients (Deepseek, Qwen) against a local
//! mock server.
use futures::StreamExt;
use mockito::Matcher;
use serde_json::json;

use crate::{
    language_models::{llm::LLM, options::CallOptions, LLMError},
    llm::{Deepseek, DeepseekError, DeepseekModel, Qwen, QwenError},
    schemas::Message,
};

fn chat_body(content: &str) -> String {
    json!({
        "id": "1", "object": "chat.completion", "created": 1, "model": "m",
        "system_fingerprint": "fp",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": content,
                                 "reasoning_content": "thinking"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    })
    .to_string()
}

fn sse(chunks: &[&str]) -> String {
    let mut body = String::new();
    for c in chunks {
        let v = json!({"choices": [{"index": 0, "delta": {"content": c}}]});
        body.push_str(&format!("data: {v}\n\n"));
    }
    body.push_str("data: [DONE]\n\n");
    body
}

async fn collect(
    s: std::pin::Pin<
        Box<dyn futures::Stream<Item = Result<crate::schemas::StreamData, LLMError>> + Send>,
    >,
) -> String {
    s.map(|r| r.unwrap().content)
        .collect::<Vec<_>>()
        .await
        .join("")
}

// ---------------- Deepseek ----------------

fn deepseek(server: &mockito::ServerGuard) -> Deepseek {
    Deepseek::new()
        .with_api_key("sk-test")
        .with_base_url(server.url())
}

#[tokio::test]
async fn deepseek_generate_sends_payload_and_parses_response() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/chat/completions")
        .match_header("authorization", "Bearer sk-test")
        .match_body(Matcher::PartialJson(json!({
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"}
            ],
            "max_tokens": 7,
            "temperature": 0.5,
            "response_format": {"type": "json_object"}
        })))
        .with_header("content-type", "application/json")
        .with_body(chat_body("hello"))
        .create_async()
        .await;

    let llm = deepseek(&server)
        .with_json_mode(true)
        .with_options(CallOptions::new().with_max_tokens(7).with_temperature(0.5));
    let res = llm
        .generate(&[
            Message::new_system_message("be brief"),
            Message::new_human_message("hi"),
        ])
        .await
        .unwrap();
    mock.assert_async().await;
    assert_eq!(res.generation, "hello");
    assert_eq!(res.tokens.unwrap().total_tokens, 5);
}

#[tokio::test]
async fn deepseek_reasoner_includes_reasoning() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/chat/completions")
        .with_body(chat_body("42"))
        .create_async()
        .await;
    let res = deepseek(&server)
        .with_model(DeepseekModel::DeepseekReasoner.to_string())
        .with_include_reasoning(true)
        .invoke("q")
        .await
        .unwrap();
    assert_eq!(res, "Reasoning:\nthinking\n\nAnswer:\n42");
}

#[tokio::test]
async fn deepseek_maps_http_errors() {
    type IsExpected = fn(&LLMError) -> bool;
    let cases: [(usize, IsExpected); 4] = [
        (401, |e| {
            matches!(
                e,
                LLMError::DeepseekError(DeepseekError::AuthenticationError(_))
            )
        }),
        (402, |e| {
            matches!(
                e,
                LLMError::DeepseekError(DeepseekError::InsufficientBalanceError(_))
            )
        }),
        (429, |e| {
            matches!(e, LLMError::DeepseekError(DeepseekError::RateLimitError(_)))
        }),
        (503, |e| {
            matches!(
                e,
                LLMError::DeepseekError(DeepseekError::ServerOverloadedError(_))
            )
        }),
    ];
    for (status, check) in cases {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/v1/chat/completions")
            .with_status(status)
            .with_body("{}")
            .create_async()
            .await;
        let err = deepseek(&server).invoke("x").await.unwrap_err();
        assert!(check(&err), "status {status}: {err:?}");
    }
}

#[tokio::test]
async fn deepseek_stream_and_streaming_func() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/chat/completions")
        .match_body(Matcher::PartialJson(json!({"stream": true})))
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&["Hel", "lo"]))
        .expect(2)
        .create_async()
        .await;

    let llm = deepseek(&server);
    let text = collect(
        llm.stream(&[Message::new_human_message("hi")])
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(text, "Hello");

    let seen = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
    let seen2 = seen.clone();
    let llm =
        deepseek(&server).with_options(CallOptions::new().with_streaming_func(move |s: String| {
            seen2.lock().unwrap().push_str(&s);
            async { Ok(()) }
        }));
    assert_eq!(llm.invoke("hi").await.unwrap(), "Hello");
    assert_eq!(*seen.lock().unwrap(), "Hello");
}

// ---------------- Qwen ----------------

fn qwen(server: &mockito::ServerGuard) -> Qwen {
    Qwen::new()
        .with_api_key("qk")
        .with_base_url(format!("{}/chat", server.url()))
}

#[tokio::test]
async fn qwen_generate_and_errors() {
    let mut server = mockito::Server::new_async().await;
    let ok = server
        .mock("POST", "/chat")
        .match_header("authorization", "Bearer qk")
        .match_body(Matcher::PartialJson(json!({
            "messages": [{"role": "user", "content": "hi"}]
        })))
        .with_body(chat_body("ni hao"))
        .create_async()
        .await;
    let res = qwen(&server)
        .generate(&[Message::new_human_message("hi")])
        .await
        .unwrap();
    ok.assert_async().await;
    assert_eq!(res.generation, "ni hao");
    assert_eq!(res.tokens.unwrap().prompt_tokens, 3);

    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat")
        .with_status(401)
        .with_body(
            json!({"request_id": "r", "code": "InvalidApiKey", "message": "bad"}).to_string(),
        )
        .create_async()
        .await;
    let err = qwen(&server).invoke("x").await.unwrap_err();
    assert!(
        matches!(&err, LLMError::QwenError(QwenError::InvalidApiKeyError(m)) if m == "bad"),
        "{err:?}"
    );

    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat")
        .with_body(
            json!({"id": "1", "created": 1, "model": "m", "choices": [],
                   "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
            .to_string(),
        )
        .create_async()
        .await;
    assert!(matches!(
        qwen(&server).invoke("x").await.unwrap_err(),
        LLMError::ContentNotFound(_)
    ));
}

#[tokio::test]
async fn qwen_stream() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/chat")
        .match_header("accept", "text/event-stream")
        .match_body(Matcher::PartialJson(json!({"stream": true})))
        .with_header("content-type", "text/event-stream")
        .with_body(sse(&["a", "b", "c"]))
        .create_async()
        .await;
    let text = collect(
        qwen(&server)
            .stream(&[Message::new_human_message("hi")])
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(text, "abc");
}
