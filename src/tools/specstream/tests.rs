use super::*;
use mockito::Matcher;

fn keyed(server: &mockito::Server) -> SpecStreamSearch {
    SpecStreamSearch::new()
        .with_api_key("test-key")
        .with_base_url(server.url())
}

fn query_matcher(q: &str, num: &str) -> Matcher {
    Matcher::AllOf(vec![
        Matcher::UrlEncoded("q".into(), q.into()),
        Matcher::UrlEncoded("num".into(), num.into()),
        Matcher::UrlEncoded("speed".into(), "fast".into()),
        Matcher::UrlEncoded("passages".into(), "true".into()),
    ])
}

fn results_body() -> String {
    json!({
        "data": {
            "query": "tokio",
            "sources": [
                {
                    "title": "Releases · tokio-rs/tokio",
                    "url": "https://github.com/tokio-rs/tokio/releases",
                    "snippet": "Tokio is a runtime for writing reliable applications.",
                    "highlights": ["  runtime: stabilize `worker_index()`  ", " ", "net: add `UdpSocket::peek`"],
                    "published": "2026-11-21"
                },
                {
                    "title": "",
                    "url": "https://example.com/b",
                    "snippet": "A page about b."
                },
                {
                    "url": "https://example.com/c",
                    "highlights": []
                }
            ]
        },
        "error": null
    })
    .to_string()
}

#[tokio::test]
async fn search_sends_bearer_key_and_formats_passages() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("GET", "/v1/web-search")
        .match_header("authorization", "Bearer test-key")
        .match_query(query_matcher("tokio", "5"))
        .with_header("content-type", "application/json")
        .with_body(results_body())
        .create_async()
        .await;

    let out = keyed(&server).run(json!("tokio")).await.unwrap();
    mock.assert_async().await;
    assert_eq!(
        out,
        "[1] Title: Releases · tokio-rs/tokio\n\
         URL: https://github.com/tokio-rs/tokio/releases\n\
         Published: 2026-11-21\n\
         Passages: runtime: stabilize `worker_index()` ... net: add `UdpSocket::peek`\n\n\
         [2] Title: (untitled)\n\
         URL: https://example.com/b\n\
         Snippet: A page about b.\n\n\
         [3] Title: (untitled)\n\
         URL: https://example.com/c"
    );
}

#[tokio::test]
async fn typed_results_and_input_field() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("GET", "/v1/web-search")
        .match_query(query_matcher("rust", "2"))
        .with_header("content-type", "application/json")
        .with_body(results_body())
        .expect(2)
        .create_async()
        .await;

    let tool = keyed(&server).with_max_results(2).unwrap();
    let results = tool.search("rust").await.unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].published.as_deref(), Some("2026-11-21"));
    assert_eq!(results[0].highlights.len(), 3);
    let input = tool.parse_input(r#"{"input": "rust"}"#).await;
    tool.run(input).await.unwrap();
    mock.assert_async().await;
}

#[tokio::test]
async fn empty_sources_give_a_plain_message() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/v1/web-search")
        .match_query(Matcher::Any)
        .with_header("content-type", "application/json")
        .with_body(json!({"data": {"sources": []}, "error": null}).to_string())
        .create_async()
        .await;

    let out = keyed(&server).run(json!({"query": "rust"})).await.unwrap();
    assert_eq!(out, "No results found for query: rust");
}

#[tokio::test]
async fn http_error_carries_status_and_api_message() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/v1/web-search")
        .match_query(Matcher::Any)
        .with_status(401)
        .with_body(
            json!({"error": {"code": "invalid_token", "message": "key not accepted", "details": {}}})
                .to_string(),
        )
        .create_async()
        .await;
    server
        .mock("GET", "/v1/web-search")
        .match_query(query_matcher("slow", "5"))
        .with_status(429)
        .with_body("slow down")
        .create_async()
        .await;

    let err = keyed(&server).search("rust").await.unwrap_err().to_string();
    assert!(err.contains("401"), "{err}");
    assert!(err.contains("invalid_token: key not accepted"), "{err}");
    let err = keyed(&server).search("slow").await.unwrap_err().to_string();
    assert!(err.contains("429") && err.contains("slow down"), "{err}");
}

#[tokio::test]
async fn no_key_sends_no_request() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("GET", "/v1/web-search")
        .match_query(Matcher::Any)
        .expect(0)
        .create_async()
        .await;

    let err = SpecStreamSearch::new()
        .with_api_key("")
        .with_base_url(server.url())
        .search("rust")
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("SPECSTREAM_API_KEY"), "{err}");
    mock.assert_async().await;
}

#[tokio::test]
async fn rejects_bad_input() {
    assert!(SpecStreamSearch::new().with_max_results(0).is_err());
    assert!(SpecStreamSearch::new().with_max_results(101).is_err());
    assert!(SpecStreamSearch::new().with_max_results(100).is_ok());
    assert!(SpecStreamSearch::new().search("  ").await.is_err());
    assert!(SpecStreamSearch::new().run(json!(42)).await.is_err());
}

/// Hits the real API: `SPECSTREAM_API_KEY=... cargo test specstream -- --ignored`.
#[tokio::test]
#[ignore]
async fn live_search() {
    let results = SpecStreamSearch::new()
        .with_max_results(3)
        .unwrap()
        .search("tokio rust async runtime")
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().all(|r| r.url.starts_with("http")));
}
