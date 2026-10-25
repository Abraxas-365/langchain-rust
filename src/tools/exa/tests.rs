use super::*;
use mockito::Matcher;

fn tool(server: &mockito::Server) -> ExaSearch {
    ExaSearch::new()
        .with_api_key("test-key")
        .with_base_url(server.url())
}

fn results_body() -> String {
    json!({
        "requestId": "r1",
        "results": [
            {
                "id": "https://blog.rust-lang.org/a",
                "title": "Announcing Rust 1.85",
                "url": "https://blog.rust-lang.org/a",
                "publishedDate": "2025-02-20T00:00:00.000Z",
                "author": "The Rust Release Team",
                "highlights": ["  The 2024 edition is stable.  ", "", "Async closures too."],
                "highlightScores": [0.9, 0.1, 0.8]
            },
            {
                "id": "b",
                "title": "",
                "url": "https://example.com/b",
                "publishedDate": null,
                "author": "",
                "text": "\n Full page text. \n"
            },
            {
                "url": "https://example.com/c"
            }
        ],
        "notices": [{"code": "RESTRICTIONS_RELAXED", "path": "/query", "message": "site: matched nothing, so it was dropped."}],
        "warnings": [{"code": "IGNORED_FIELD", "message": "not for the model"}],
        "costDollars": {"total": 0.005}
    })
    .to_string()
}

fn empty_body() -> String {
    json!({"requestId": "r", "results": []}).to_string()
}

#[tokio::test]
async fn sends_defaults_with_api_key_header() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/search")
        .match_header("x-api-key", "test-key")
        .match_header("content-type", "application/json")
        .match_body(Matcher::Json(json!({
            "query": "rust 2024",
            "numResults": 5,
            "contents": {"highlights": true}
        })))
        .with_header("content-type", "application/json")
        .with_body(results_body())
        .create_async()
        .await;

    let out = tool(&server).run(json!("rust 2024")).await.unwrap();
    mock.assert_async().await;
    assert_eq!(
        out,
        "[1] Title: Announcing Rust 1.85\n\
         URL: https://blog.rust-lang.org/a\n\
         Published: 2025-02-20\n\
         Author: The Rust Release Team\n\
         Highlights:\n\
         - The 2024 edition is stable.\n\
         - Async closures too.\n\n\
         [2] Title: (untitled)\n\
         URL: https://example.com/b\n\
         Text: Full page text.\n\n\
         [3] Title: (untitled)\n\
         URL: https://example.com/c\n\n\
         Notes from the search engine:\n\
         - site: matched nothing, so it was dropped."
    );
    assert!(
        !out.contains("not for the model"),
        "warnings are logged, not shown"
    );
}

#[tokio::test]
async fn sends_every_option() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/search")
        .match_body(Matcher::Json(json!({
            "query": "rate decisions",
            "numResults": 2,
            "type": "deep",
            "category": "news",
            "includeDomains": ["reuters.com", "*.apnews.com"],
            "excludeDomains": ["example.com/spam"],
            "startPublishedDate": "2026-01-01",
            "endPublishedDate": "2026-02-01T00:00:00Z",
            "contents": {"text": {"maxCharacters": 500}}
        })))
        .with_header("content-type", "application/json")
        .with_body(empty_body())
        .create_async()
        .await;

    let exa = tool(&server)
        .with_num_results(2)
        .unwrap()
        .with_search_type("deep")
        .with_category("news")
        .with_include_domains(["reuters.com", "*.apnews.com"])
        .with_exclude_domains(["example.com/spam"])
        .with_start_published_date("2026-01-01")
        .unwrap()
        .with_end_published_date("2026-02-01T00:00:00Z")
        .unwrap()
        .with_contents(ExaContents::Text {
            max_characters: Some(500),
        })
        .unwrap();
    let out = exa.run(json!("rate decisions")).await.unwrap();
    mock.assert_async().await;
    assert_eq!(out, "No results found for query: rate decisions");
}

#[test]
fn contents_options_serialize() {
    assert_eq!(ExaContents::None.to_json(), None);
    assert_eq!(
        ExaContents::default().to_json(),
        Some(json!({"highlights": true}))
    );
    assert_eq!(
        ExaContents::Highlights {
            max_characters: Some(800)
        }
        .to_json(),
        Some(json!({"highlights": {"maxCharacters": 800}}))
    );
    assert_eq!(
        ExaContents::Text {
            max_characters: None
        }
        .to_json(),
        Some(json!({"text": true}))
    );
}

#[tokio::test]
async fn no_contents_sends_no_contents_field() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/search")
        .match_body(Matcher::Json(json!({"query": "q", "numResults": 5})))
        .with_header("content-type", "application/json")
        .with_body(empty_body())
        .create_async()
        .await;
    tool(&server)
        .with_contents(ExaContents::None)
        .unwrap()
        .run(json!("q"))
        .await
        .unwrap();
    mock.assert_async().await;
}

#[tokio::test]
async fn search_returns_typed_results_and_notices() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/search")
        .with_header("content-type", "application/json")
        .with_body(results_body())
        .create_async()
        .await;

    let search = tool(&server).search("rust").await.unwrap();
    assert_eq!(search.results.len(), 3);
    let first = &search.results[0];
    assert_eq!(first.id, "https://blog.rust-lang.org/a");
    assert_eq!(first.author.as_deref(), Some("The Rust Release Team"));
    assert_eq!(first.highlights.len(), 3);
    assert_eq!(
        search.results[1].text.as_deref(),
        Some("\n Full page text. \n")
    );
    assert_eq!(search.results[2].title, None);
    assert_eq!(search.results[2].id, "");
    assert_eq!(
        search.notices,
        ["site: matched nothing, so it was dropped."]
    );
}

#[tokio::test]
async fn notices_are_shown_even_without_results() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/search")
        .with_header("content-type", "application/json")
        .with_body(
            json!({"requestId": "r", "results": [], "notices": [
                {"code": "X", "message": "Try fewer filters."},
                {"code": "Y", "message": ""}
            ]})
            .to_string(),
        )
        .create_async()
        .await;
    let out = tool(&server).run(json!("q")).await.unwrap();
    assert_eq!(
        out,
        "No results found for query: q\n\nNotes from the search engine:\n- Try fewer filters."
    );
}

#[tokio::test]
async fn accepts_tool_call_and_input_objects() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/search")
        .match_body(Matcher::PartialJson(json!({"query": "q1"})))
        .with_header("content-type", "application/json")
        .with_body(empty_body())
        .expect(3)
        .create_async()
        .await;

    let exa = tool(&server);
    for raw in [r#"{"query": "q1"}"#, r#"{"input": "q1"}"#, "q1"] {
        let out = exa.call(raw).await.unwrap();
        assert_eq!(out, "No results found for query: q1", "{raw}");
    }
    mock.assert_async().await;
}

#[tokio::test]
async fn api_errors_carry_status_and_body() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/search")
        .with_status(402)
        .with_body(format!(
            "{{\"error\": \"out of credits\", \"tag\": \"NO_MORE_CREDITS\"}}{}",
            "x".repeat(2000)
        ))
        .create_async()
        .await;

    let err = tool(&server).run(json!("q")).await.unwrap_err().to_string();
    assert!(err.contains("402"), "{err}");
    assert!(err.contains("out of credits"), "{err}");
    assert!(err.len() < 600, "error body is cut: {} chars", err.len());
}

#[tokio::test]
async fn malformed_response_is_an_error() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/search")
        .with_header("content-type", "application/json")
        .with_body(r#"{"output": "synthesis only"}"#)
        .create_async()
        .await;
    assert!(tool(&server).run(json!("q")).await.is_err());
}

#[tokio::test]
async fn times_out() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/search")
        .with_chunked_body(|w| {
            std::thread::sleep(Duration::from_millis(1500));
            w.write_all(b"{}")
        })
        .create_async()
        .await;
    let exa = tool(&server).with_timeout(Duration::from_millis(200));
    let started = std::time::Instant::now();
    let result = tokio::time::timeout(Duration::from_secs(5), exa.run(json!("q")))
        .await
        .expect("client timeout should fire first");
    assert!(result.is_err());
    assert!(started.elapsed() < Duration::from_millis(1400));
}

#[tokio::test]
async fn rejects_bad_input_before_any_request() {
    // Nothing listens on port 1; these must fail before sending.
    let exa = ExaSearch::new()
        .with_api_key("k")
        .with_base_url("http://127.0.0.1:1");
    for input in [
        json!(""),
        json!("  "),
        json!(42),
        json!({"q": "x"}),
        json!({"query": 1}),
    ] {
        let err = exa.run(input.clone()).await.unwrap_err().to_string();
        assert!(
            err.contains("empty") || err.contains("expected a string"),
            "{input}: {err}"
        );
    }
    let err = ExaSearch::new()
        .with_api_key("")
        .with_base_url("http://127.0.0.1:1")
        .run(json!("q"))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("EXA_API_KEY"), "{err}");
}

#[test]
fn option_validation() {
    let s = ExaSearch::new;
    assert!(s().with_num_results(0).is_err());
    assert!(s().with_num_results(101).is_err());
    assert_eq!(s().with_num_results(1).unwrap().num_results, 1);
    assert_eq!(s().with_num_results(100).unwrap().num_results, 100);
    for ok in [
        "2026-01-31",
        "2026-01-31T10:00:00Z",
        "2026-01-31T10:00:00.000+05:00",
    ] {
        assert!(s().with_start_published_date(ok).is_ok(), "{ok}");
        assert!(s().with_end_published_date(ok).is_ok(), "{ok}");
    }
    for bad in [
        "",
        "2026",
        "2026-1-31",
        "31/01/2026",
        "2026-01-31 10:00",
        "2026/01/31",
        "20260131xx",
        "2026-01-3a",
        "a026-01-31",
        "2026-a1-31",
        "2026x01-31",
        "2026-01x31",
    ] {
        assert!(s().with_start_published_date(bad).is_err(), "{bad}");
        assert!(s().with_end_published_date(bad).is_err(), "{bad}");
    }
    for zero in [
        ExaContents::Highlights {
            max_characters: Some(0),
        },
        ExaContents::Text {
            max_characters: Some(0),
        },
    ] {
        assert!(s().with_contents(zero).is_err());
    }
    assert_eq!(s().with_base_url("http://x/").base_url, "http://x");
}

#[test]
fn tool_metadata() {
    let exa = ExaSearch::new();
    assert_eq!(exa.name(), "exa-search");
    assert_eq!(exa.parameters()["required"], json!(["query"]));
    assert!(exa.description().contains("search query"));
}

#[tokio::test]
async fn agent_reads_search_results() {
    use crate::{
        agent::{AgentExecutor, OpenAiToolAgentBuilder},
        chain::Chain,
        prompt_args,
        schemas::{FunctionCallResponse, FunctionDetail, MessageType},
        test_utils::FakeLLM,
    };
    use std::sync::Arc;

    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/search")
        .match_body(Matcher::PartialJson(json!({"query": "rust 1.85"})))
        .with_header("content-type", "application/json")
        .with_body(results_body())
        .create_async()
        .await;

    let call = FunctionCallResponse {
        id: "call_1".into(),
        type_field: "function".into(),
        function: FunctionDetail {
            name: "exa-search".into(),
            arguments: r#"{"query": "rust 1.85"}"#.into(),
        },
    };
    let llm = FakeLLM::new([
        serde_json::to_string(&vec![call]).unwrap(),
        "It stabilized the 2024 edition.".to_string(),
    ]);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(tool(&server))])
        .build(llm.clone())
        .unwrap();
    let out = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "what did rust 1.85 ship?"})
        .await
        .unwrap();
    assert_eq!(out, "It stabilized the 2024 edition.");
    mock.assert_async().await;

    let observation = llm.calls()[1]
        .iter()
        .find(|m| m.message_type == MessageType::ToolMessage)
        .expect("search results go back to the model")
        .content
        .clone();
    assert!(observation.starts_with("[1] Title: Announcing Rust 1.85\n"));
    assert!(observation.contains("- The 2024 edition is stable."));
}

/// A real model against a mocked Exa API.
/// `OLLAMA_MODEL=qwen3:14b cargo test --lib exa_agent_live -- --ignored --nocapture`
#[tokio::test]
#[ignore]
async fn exa_agent_live_ollama() {
    use crate::{
        agent::{AgentExecutor, OpenAiToolAgentBuilder},
        chain::Chain,
        llm::{ollama::openai::OllamaConfig, openai::OpenAI},
        prompt_args,
    };
    use std::sync::Arc;

    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/search")
        .match_body(Matcher::Regex("(?i)quuxian".into()))
        .with_header("content-type", "application/json")
        .with_body(
            json!({"requestId": "r", "results": [{
                "id": "1",
                "title": "Quuxian Republic adopts a new capital",
                "url": "https://example.com/quux",
                "publishedDate": "2031-05-01T00:00:00.000Z",
                "highlights": ["The Quuxian Republic moved its capital to Port Vellan in May 2031."]
            }]})
            .to_string(),
        )
        .expect_at_least(1)
        .create_async()
        .await;

    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:14b".into());
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(tool(&server))])
        .build(OpenAI::new(OllamaConfig::default()).with_model(&model))
        .unwrap();
    let out = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {
            "input" => "Search the web: what is the current capital of the Quuxian Republic? Answer in one sentence."
        })
        .await
        .unwrap();
    println!("{out}");
    mock.assert_async().await;
    assert!(out.contains("Port Vellan"), "{out}");
}

/// `EXA_API_KEY=... cargo test --lib exa_live -- --ignored`
#[tokio::test]
#[ignore]
async fn exa_live() {
    if std::env::var("EXA_API_KEY").is_err() {
        eprintln!("EXA_API_KEY not set, skipping");
        return;
    }
    let search = ExaSearch::new()
        .with_num_results(3)
        .unwrap()
        .with_contents(ExaContents::Highlights {
            max_characters: Some(300),
        })
        .unwrap()
        .search("Rust programming language")
        .await
        .unwrap();
    assert!(!search.results.is_empty());
    assert!(search.results.iter().all(|r| r.url.starts_with("http")));
}
