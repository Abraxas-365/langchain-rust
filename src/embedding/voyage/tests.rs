use std::sync::{Arc, Mutex};

use mockito::{Matcher, Server};
use serde_json::{json, Value};

use super::*;
use crate::embedding::Embedder;

/// Answers each text with `[number in "doc N"]`, listing `data` in reverse order so the
/// embedder has to sort by `index`.
fn reply(request: &mockito::Request) -> Vec<u8> {
    let body: Value = serde_json::from_slice(request.body().unwrap()).unwrap();
    let mut data: Vec<Value> = body["input"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, text)| {
            let n: f64 = text.as_str().unwrap()[4..].parse().unwrap();
            json!({ "object": "embedding", "embedding": [n], "index": index })
        })
        .collect();
    data.reverse();
    json!({ "object": "list", "data": data, "model": body["model"], "usage": { "total_tokens": 1 } })
        .to_string()
        .into_bytes()
}

fn docs(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("doc {i}")).collect()
}

#[tokio::test]
async fn documents_are_batched_ordered_and_marked_as_documents() {
    let mut server = Server::new_async().await;
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let seen = sizes.clone();
    let mock = server
        .mock("POST", "/embeddings")
        .match_header("authorization", "Bearer vk-test")
        .match_header("content-type", "application/json")
        .match_body(Matcher::PartialJson(json!({
            "model": "voyage-3.5-lite",
            "input_type": "document",
            "output_dimension": 512,
            "truncation": false,
        })))
        .with_body_from_request(move |request| {
            let body: Value = serde_json::from_slice(request.body().unwrap()).unwrap();
            seen.lock()
                .unwrap()
                .push(body["input"].as_array().unwrap().len());
            reply(request)
        })
        .expect(3)
        .create_async()
        .await;

    let embedder = VoyageEmbedder::new()
        .with_base_url(format!("{}/", server.url()))
        .with_api_key("vk-test")
        .with_model("voyage-3.5-lite")
        .with_output_dimension(512)
        .with_truncation(false)
        .with_batch_size(100);
    let vectors = embedder.embed_documents(&docs(250)).await.unwrap();
    mock.assert_async().await;
    assert_eq!(*sizes.lock().unwrap(), [100, 100, 50]);
    assert_eq!(
        vectors,
        (0..250).map(|i| vec![i as f64]).collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn query_uses_query_input_type_and_defaults() {
    let mut server = Server::new_async().await;
    let mock = server
        .mock("POST", "/embeddings")
        .match_body(Matcher::Json(json!({
            "input": ["doc 7"], "model": "voyage-3.5", "input_type": "query",
        })))
        .with_body_from_request(reply)
        .create_async()
        .await;
    let embedder = VoyageEmbedder::new()
        .with_base_url(server.url())
        .with_api_key("k");
    assert_eq!(embedder.embed_query("doc 7").await.unwrap(), [7.0]);
    mock.assert_async().await;
}

#[tokio::test]
async fn default_batch_size_is_128_and_is_clamped() {
    let mut server = Server::new_async().await;
    let sizes = Arc::new(Mutex::new(Vec::new()));
    let seen = sizes.clone();
    server
        .mock("POST", "/embeddings")
        .with_body_from_request(move |request| {
            let body: Value = serde_json::from_slice(request.body().unwrap()).unwrap();
            seen.lock()
                .unwrap()
                .push(body["input"].as_array().unwrap().len());
            reply(request)
        })
        .create_async()
        .await;
    let base = || {
        VoyageEmbedder::new()
            .with_base_url(server.url())
            .with_api_key("k")
    };

    base().embed_documents(&docs(130)).await.unwrap();
    base()
        .with_batch_size(0)
        .embed_documents(&docs(2))
        .await
        .unwrap();
    base()
        .with_batch_size(5000)
        .embed_documents(&docs(1001))
        .await
        .unwrap();
    assert_eq!(*sizes.lock().unwrap(), [128, 2, 1, 1, 1000, 1]);

    // No documents, no request.
    sizes.lock().unwrap().clear();
    assert!(base().embed_documents(&[]).await.unwrap().is_empty());
    assert!(sizes.lock().unwrap().is_empty());
}

#[tokio::test]
async fn api_errors_are_reported() {
    let mut server = Server::new_async().await;
    let embedder = VoyageEmbedder::new()
        .with_base_url(server.url())
        .with_api_key("bad");

    let mock = server
        .mock("POST", "/embeddings")
        .with_status(401)
        .with_body(r#"{"detail":"Provided API key is invalid."}"#)
        .create_async()
        .await;
    assert_eq!(
        embedder.embed_query("doc 1").await.unwrap_err().to_string(),
        "HTTP error: 401 Unauthorized Provided API key is invalid."
    );
    mock.remove_async().await;

    let mock = server
        .mock("POST", "/embeddings")
        .with_status(502)
        .with_body("<html>bad gateway</html>")
        .create_async()
        .await;
    assert_eq!(
        embedder.embed_query("doc 1").await.unwrap_err().to_string(),
        "HTTP error: 502 Bad Gateway <html>bad gateway</html>"
    );
    mock.remove_async().await;

    let mock = server
        .mock("POST", "/embeddings")
        .with_body(r#"{"data":[]}"#)
        .create_async()
        .await;
    assert_eq!(
        embedder.embed_query("doc 1").await.unwrap_err().to_string(),
        "HTTP error: 200 OK Voyage returned 0 embeddings for 1 texts"
    );
    mock.remove_async().await;

    server
        .mock("POST", "/embeddings")
        .with_body(r#"{"data":"nope"}"#)
        .create_async()
        .await;
    let err = embedder.embed_query("doc 1").await.unwrap_err().to_string();
    assert!(err.contains("invalid Voyage response"), "{err}");
}

/// `VOYAGE_API_KEY=... cargo test -- --ignored voyage_live`
#[tokio::test]
#[ignore = "needs VOYAGE_API_KEY"]
async fn voyage_live() {
    assert!(std::env::var("VOYAGE_API_KEY").is_ok());
    let embedder = VoyageEmbedder::new()
        .with_model("voyage-3.5-lite")
        .with_output_dimension(256);
    let vectors = embedder
        .embed_documents(&["hello".into(), "world".into()])
        .await
        .unwrap();
    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0].len(), 256);
    assert_eq!(embedder.embed_query("hi").await.unwrap().len(), 256);
}
