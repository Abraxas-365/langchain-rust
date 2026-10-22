//! Offline tests: a real Bedrock SDK client pointed at a mockito server.

use std::sync::{Arc, Mutex};

use aws_sdk_bedrockruntime::config::{retry::RetryConfig, Credentials, Region};
use aws_sdk_bedrockruntime::Client as BedrockClient;
use mockito::{Matcher, Server, ServerGuard};
use serde_json::{json, Value};

use super::bedrock_embedder::unexpected;
use super::*;
use crate::embedding::Embedder;

fn client_for(server: &ServerGuard) -> BedrockClient {
    client_at(&server.url())
}

/// `/model/{id}/invoke`, with the `:` of versioned ids possibly percent-encoded.
fn invoke_path(model_id: &str) -> Matcher {
    let id = regex::escape(model_id).replace(':', "(:|%3A)");
    Matcher::Regex(format!("^/model/{id}/invoke$"))
}

fn json_body(request: &mockito::Request) -> Value {
    serde_json::from_slice(request.body().unwrap()).unwrap()
}

#[tokio::test]
async fn titan_query_sends_input_text_and_options() {
    let mut server = Server::new_async().await;
    let mock = server
        .mock("POST", invoke_path("amazon.titan-embed-text-v2:0"))
        .match_header("content-type", "application/json")
        .match_header("accept", "application/json")
        .match_body(Matcher::Json(
            json!({ "inputText": "hello", "dimensions": 256, "normalize": false }),
        ))
        .with_body(r#"{"embedding":[0.25,-0.5,1],"inputTextTokenCount":2}"#)
        .create_async()
        .await;

    let embedder = BedrockEmbedder::new()
        .with_client(client_for(&server))
        .with_dimensions(256)
        .with_normalize(false);
    assert_eq!(
        embedder.embed_query("hello").await.unwrap(),
        [0.25, -0.5, 1.0]
    );
    mock.assert_async().await;
}

#[tokio::test]
async fn titan_v1_sends_only_the_text() {
    let mut server = Server::new_async().await;
    let mock = server
        .mock("POST", invoke_path("amazon.titan-embed-text-v1"))
        .match_body(Matcher::Json(json!({ "inputText": "hello" })))
        .with_body(r#"{"embedding":[1.0]}"#)
        .create_async()
        .await;
    let embedder = BedrockEmbedder::new()
        .with_client(client_for(&server))
        .with_model(BedrockEmbeddingModel::TitanEmbedTextV1);
    assert_eq!(embedder.embed_query("hello").await.unwrap(), [1.0]);
    mock.assert_async().await;
}

/// A concurrent Titan stand-in (mockito answers one request at a time, which can't show
/// parallelism). The vector is `[len(text), first byte of text]`; a text of `"first"` is
/// answered 150ms late, every other one after 40ms. Returns the URL and the
/// `(in flight, max in flight)` counters.
async fn concurrent_titan_server() -> (String, Arc<Mutex<(usize, usize)>>) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let counters = Arc::new(Mutex::new((0usize, 0usize)));
    let in_flight = counters.clone();
    tokio::spawn(async move {
        loop {
            let (mut socket, _) = listener.accept().await.unwrap();
            let in_flight = in_flight.clone();
            tokio::spawn(async move {
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let body = loop {
                    let n = socket.read(&mut chunk).await.unwrap();
                    if n == 0 {
                        return;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    let text = String::from_utf8_lossy(&buf).to_string();
                    let Some(end) = text.find("\r\n\r\n") else {
                        continue;
                    };
                    let length = text[..end]
                        .lines()
                        .find_map(|l| {
                            let (k, v) = l.split_once(':')?;
                            k.eq_ignore_ascii_case("content-length")
                                .then(|| v.trim().parse::<usize>().unwrap())
                        })
                        .unwrap_or(0);
                    if buf.len() >= end + 4 + length {
                        break buf[end + 4..end + 4 + length].to_vec();
                    }
                };
                let text = serde_json::from_slice::<Value>(&body).unwrap()["inputText"]
                    .as_str()
                    .unwrap()
                    .to_string();
                {
                    let mut c = in_flight.lock().unwrap();
                    c.0 += 1;
                    c.1 = c.1.max(c.0);
                }
                let delay = if text == "first" { 150 } else { 40 };
                tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
                in_flight.lock().unwrap().0 -= 1;
                let reply = json!({ "embedding": [text.len() as f64, text.as_bytes()[0] as f64] })
                    .to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
                    reply.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
                socket.shutdown().await.ok();
            });
        }
    });
    (url, counters)
}

fn client_at(url: &str) -> BedrockClient {
    let conf = aws_sdk_bedrockruntime::Config::builder()
        .behavior_version_latest()
        .region(Region::new("us-east-1"))
        .endpoint_url(url)
        .credentials_provider(Credentials::new("AKID", "SECRET", None, None, "test"))
        .retry_config(RetryConfig::disabled())
        .build();
    BedrockClient::from_conf(conf)
}

#[tokio::test]
async fn titan_documents_are_embedded_concurrently_and_keep_their_order() {
    let (url, counters) = concurrent_titan_server().await;
    let embedder = BedrockEmbedder::new()
        .with_client(client_at(&url))
        .with_concurrency(3);
    // "first" answers last, so the order can't come from completion order.
    let docs = ["first", "second", "x"].map(String::from);
    assert_eq!(
        embedder.embed_documents(&docs).await.unwrap(),
        [vec![5.0, 102.0], vec![6.0, 115.0], vec![1.0, 120.0]]
    );
    assert_eq!(
        counters.lock().unwrap().1,
        3,
        "all three were in flight at once"
    );
}

#[tokio::test]
async fn titan_concurrency_bounds_requests_in_flight() {
    let (url, counters) = concurrent_titan_server().await;
    let docs: Vec<String> = (0..6).map(|i| format!("doc {i}")).collect();
    for (concurrency, expected_max) in [(2, 2), (1, 1), (0, 1)] {
        *counters.lock().unwrap() = (0, 0);
        let embedder = BedrockEmbedder::new()
            .with_client(client_at(&url))
            .with_concurrency(concurrency);
        // `buffered(0)` would never poll a request, so guard against a hang.
        let vectors = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            embedder.embed_documents(&docs),
        )
        .await
        .expect("embed_documents hung")
        .unwrap();
        assert_eq!(vectors.len(), 6);
        assert_eq!(
            counters.lock().unwrap().1,
            expected_max,
            "concurrency {concurrency}"
        );
    }
    // The default is 5.
    *counters.lock().unwrap() = (0, 0);
    let docs: Vec<String> = (0..8).map(|i| format!("doc {i}")).collect();
    BedrockEmbedder::new()
        .with_client(client_at(&url))
        .embed_documents(&docs)
        .await
        .unwrap();
    assert_eq!(counters.lock().unwrap().1, 5);
}

#[tokio::test]
async fn no_documents_means_no_requests() {
    let mut server = Server::new_async().await;
    let mock = server
        .mock("POST", Matcher::Any)
        .expect(0)
        .create_async()
        .await;
    for model in [
        BedrockEmbeddingModel::TitanEmbedTextV2,
        BedrockEmbeddingModel::CohereEmbedEnglishV3,
    ] {
        let embedder = BedrockEmbedder::new()
            .with_client(client_for(&server))
            .with_model(model);
        assert!(embedder.embed_documents(&[]).await.unwrap().is_empty());
    }
    mock.assert_async().await;
}

#[tokio::test]
async fn cohere_batches_documents_and_marks_queries() {
    let mut server = Server::new_async().await;
    let batches = Arc::new(Mutex::new(Vec::<usize>::new()));
    let seen = batches.clone();
    let documents = server
        .mock("POST", invoke_path("cohere.embed-multilingual-v3"))
        .match_body(Matcher::PartialJson(
            json!({ "input_type": "search_document", "truncate": "END" }),
        ))
        .with_body_from_request(move |request| {
            let texts: Vec<String> =
                serde_json::from_value(json_body(request)["texts"].clone()).unwrap();
            seen.lock().unwrap().push(texts.len());
            // Each vector is the document number, parsed from "doc N".
            let embeddings: Vec<Vec<f64>> = texts
                .iter()
                .map(|t| vec![t[4..].parse::<f64>().unwrap()])
                .collect();
            json!({ "id": "x", "texts": texts, "embeddings": embeddings })
                .to_string()
                .into_bytes()
        })
        .expect(3)
        .create_async()
        .await;
    let query = server
        .mock("POST", invoke_path("cohere.embed-multilingual-v3"))
        .match_body(Matcher::Json(json!({
            "texts": ["what?"], "input_type": "search_query", "truncate": "END",
        })))
        // Typed responses (embedding_types / Embed v4) nest the vectors under `float`.
        .with_body(r#"{"embeddings":{"float":[[0.5,0.5]]}}"#)
        .create_async()
        .await;

    let embedder = BedrockEmbedder::new()
        .with_client(client_for(&server))
        .with_model(BedrockEmbeddingModel::CohereEmbedMultilingualV3)
        .with_truncate("END");
    let docs: Vec<String> = (0..200).map(|i| format!("doc {i}")).collect();
    let vectors = embedder.embed_documents(&docs).await.unwrap();
    assert_eq!(*batches.lock().unwrap(), [96, 96, 8]);
    assert_eq!(
        vectors,
        (0..200).map(|i| vec![i as f64]).collect::<Vec<_>>()
    );
    assert_eq!(embedder.embed_query("what?").await.unwrap(), [0.5, 0.5]);
    documents.assert_async().await;
    query.assert_async().await;
}

#[tokio::test]
async fn custom_model_ids_pick_the_format_by_provider() {
    let mut server = Server::new_async().await;
    let cohere = server
        .mock("POST", invoke_path("us.cohere.embed-v4:0"))
        .match_body(Matcher::Json(
            json!({ "texts": ["q"], "input_type": "search_query" }),
        ))
        .with_body(r#"{"embeddings":[[3.0]]}"#)
        .create_async()
        .await;
    let titan = server
        .mock("POST", invoke_path("my-titan-profile"))
        .match_body(Matcher::Json(json!({ "inputText": "q" })))
        .with_body(r#"{"embedding":[4.0]}"#)
        .create_async()
        .await;
    let embed = |id: &str| {
        BedrockEmbedder::new()
            .with_client(client_for(&server))
            .with_model(BedrockEmbeddingModel::Custom(id.into()))
    };
    assert_eq!(
        embed("us.cohere.embed-v4:0")
            .embed_query("q")
            .await
            .unwrap(),
        [3.0]
    );
    assert_eq!(
        embed("my-titan-profile").embed_query("q").await.unwrap(),
        [4.0]
    );
    cohere.assert_async().await;
    titan.assert_async().await;
}

#[tokio::test]
async fn errors_are_reported() {
    let mut server = Server::new_async().await;
    let embedder = BedrockEmbedder::new().with_client(client_for(&server));

    let mock = server
        .mock("POST", Matcher::Any)
        .with_status(400)
        .with_header("x-amzn-errortype", "ValidationException")
        .with_body(r#"{"message":"Malformed input request: expected maxLength: 50000"}"#)
        .create_async()
        .await;
    let err = embedder.embed_query("hello").await.unwrap_err().to_string();
    assert!(err.starts_with("Bedrock error: "), "{err}");
    assert!(err.contains("Malformed input request"), "{err}");
    mock.remove_async().await;

    let mock = server
        .mock("POST", Matcher::Any)
        .with_body(r#"{"message":"no vectors here"}"#)
        .create_async()
        .await;
    assert_eq!(
        embedder.embed_query("hello").await.unwrap_err().to_string(),
        r#"Bedrock error: InvokeModel response has no `embedding`: {"message":"no vectors here"}"#
    );
    mock.remove_async().await;

    server
        .mock("POST", Matcher::Any)
        .with_body(r#"{"embeddings":[[1.0]]}"#)
        .create_async()
        .await;
    let cohere = embedder.with_model(BedrockEmbeddingModel::CohereEmbedEnglishV3);
    assert_eq!(
        cohere
            .embed_documents(&["a".into(), "b".into()])
            .await
            .unwrap_err()
            .to_string(),
        "Bedrock error: Cohere returned 1 embeddings for 2 texts"
    );
}

#[test]
fn long_unexpected_responses_are_truncated() {
    // `{"message":"a` is 13 bytes, so byte 200 falls inside a two-byte `é`.
    let response = json!({ "message": format!("a{}", "é".repeat(300)) });
    let err = unexpected("embedding", &response).to_string();
    assert!(err.ends_with("é..."), "{err}");
    assert!(err.len() < 300, "{}", err.len());
}

/// `BEDROCK_EMBED_LIVE=1 cargo test --features bedrock -- --ignored bedrock_embed_live`
/// with AWS credentials and access to Titan V2 in the default region.
#[tokio::test]
#[ignore = "needs AWS credentials (BEDROCK_EMBED_LIVE=1)"]
async fn bedrock_embed_live() {
    assert!(std::env::var("BEDROCK_EMBED_LIVE").is_ok());
    let embedder = BedrockEmbedder::new().with_dimensions(256);
    let docs = embedder
        .embed_documents(&["hello".into(), "world".into()])
        .await
        .unwrap();
    assert_eq!(docs.len(), 2);
    assert_eq!(docs[0].len(), 256);
    let norm: f64 = docs[0].iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "{norm}");
}
