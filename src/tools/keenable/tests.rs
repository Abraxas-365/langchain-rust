use super::*;
use mockito::Matcher;

fn keyless(server: &mockito::Server) -> KeenableSearch {
    KeenableSearch::new()
        .with_api_key("")
        .with_base_url(server.url())
}

fn results_body() -> String {
    json!({
        "query": "tokio",
        "mode": "pro",
        "results": [
            {
                "title": "Releases · tokio-rs/tokio",
                "url": "https://github.com/tokio-rs/tokio/releases",
                "description": "",
                "snippet": "  runtime: stabilize `worker_index()`  ",
                "acquired_at": "2026-11-21T15:44:43Z"
            },
            {
                "title": "",
                "url": "https://example.com/b",
                "description": "A page about b."
            },
            {
                "url": "https://example.com/c"
            }
        ]
    })
    .to_string()
}

#[tokio::test]
async fn keyless_call_uses_public_endpoint_and_title_header() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/search/public")
        .match_header("x-keenable-title", "langchain-rust")
        .match_header("x-api-key", Matcher::Missing)
        .match_body(Matcher::Json(json!({"query": "tokio", "max_results": 5})))
        .with_header("content-type", "application/json")
        .with_body(results_body())
        .create_async()
        .await;

    let out = keyless(&server).run(json!("tokio")).await.unwrap();
    mock.assert_async().await;
    assert_eq!(
        out,
        "[1] Title: Releases · tokio-rs/tokio\n\
         URL: https://github.com/tokio-rs/tokio/releases\n\
         Snippet: runtime: stabilize `worker_index()`\n\n\
         [2] Title: (untitled)\n\
         URL: https://example.com/b\n\
         Description: A page about b.\n\n\
         [3] Title: (untitled)\n\
         URL: https://example.com/c"
    );
}

#[tokio::test]
async fn api_key_uses_keyed_endpoint() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/search")
        .match_header("x-api-key", "test-key")
        .match_body(Matcher::Json(json!({"query": "rust", "max_results": 2})))
        .with_header("content-type", "application/json")
        .with_body(json!({"results": []}).to_string())
        .create_async()
        .await;

    let out = keyless(&server)
        .with_api_key("test-key")
        .with_max_results(2)
        .unwrap()
        .run(json!({"query": "rust"}))
        .await
        .unwrap();
    mock.assert_async().await;
    assert_eq!(out, "No results found for query: rust");
}

#[tokio::test]
async fn custom_app_title_and_input_field() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v1/search/public")
        .match_header("x-keenable-title", "my-agent")
        .with_header("content-type", "application/json")
        .with_body(json!({"results": []}).to_string())
        .create_async()
        .await;

    let tool = keyless(&server).with_app_title("my-agent");
    let input = tool.parse_input(r#"{"input": "rust"}"#).await;
    tool.run(input).await.unwrap();
    mock.assert_async().await;
}

#[tokio::test]
async fn http_error_includes_status_and_body() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("POST", "/v1/search/public")
        .with_status(429)
        .with_body("slow down")
        .create_async()
        .await;

    let err = keyless(&server).search("rust").await.unwrap_err();
    let err = err.to_string();
    assert!(err.contains("429"), "{err}");
    assert!(err.contains("slow down"), "{err}");
}

#[tokio::test]
async fn rejects_bad_input() {
    assert!(KeenableSearch::new().with_max_results(0).is_err());
    assert!(KeenableSearch::new().with_max_results(51).is_err());
    assert!(KeenableSearch::new().search("  ").await.is_err());
    assert!(KeenableSearch::new().run(json!(42)).await.is_err());
}

/// Hits the real API: `cargo test keenable -- --ignored`.
#[tokio::test]
#[ignore]
async fn live_keyless_search() {
    let results = KeenableSearch::new()
        .with_api_key("")
        .with_max_results(3)
        .unwrap()
        .search("tokio rust async runtime")
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().all(|r| r.url.starts_with("http")));
}
