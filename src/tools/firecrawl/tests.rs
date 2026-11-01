use super::*;
use crate::tools::Tool;
use mockito::Matcher;
use serde_json::json;

fn search_tool(server: &mockito::Server) -> FirecrawlSearch {
    FirecrawlSearch::new()
        .with_api_key("test-key")
        .with_base_url(server.url())
}

fn scrape_tool(server: &mockito::Server) -> FirecrawlScrape {
    FirecrawlScrape::new()
        .with_api_key("test-key")
        .with_base_url(server.url())
}

fn tagged(mut body: Value) -> Value {
    body["origin"] = json!("langchain-rust");
    body["integration"] = json!("_langchain-rust");
    body
}

fn search_body() -> String {
    json!({
        "success": true,
        "data": {
            "web": [
                {
                    "url": "https://blog.rust-lang.org/a",
                    "title": "Announcing Rust 1.85",
                    "description": "  The 2024 edition is stable.  ",
                    "position": 1
                },
                {
                    "url": "https://example.com/b",
                    "title": "",
                    "description": ""
                },
                {
                    "url": "https://example.com/c",
                    "markdown": "# Page C\n\nBody text.",
                    "metadata": {
                        "title": "Meta title C",
                        "description": "Meta desc C",
                        "sourceURL": "https://example.com/c",
                        "statusCode": 200
                    }
                }
            ]
        },
        "warning": "not for the model",
        "creditsUsed": 3
    })
    .to_string()
}

fn empty_search_body() -> String {
    json!({"success": true, "data": {"web": []}}).to_string()
}

#[tokio::test]
async fn search_sends_defaults_with_bearer_key() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v2/search")
        .match_header("authorization", "Bearer test-key")
        .match_header("content-type", "application/json")
        .match_body(Matcher::Json(tagged(
            json!({"query": "rust 2024", "limit": 5}),
        )))
        .with_header("content-type", "application/json")
        .with_body(search_body())
        .create_async()
        .await;

    let out = search_tool(&server).run(json!("rust 2024")).await.unwrap();
    mock.assert_async().await;
    assert_eq!(
        out,
        "[1] Title: Announcing Rust 1.85\n\
         URL: https://blog.rust-lang.org/a\n\
         Description: The 2024 edition is stable.\n\n\
         [2] Title: (untitled)\n\
         URL: https://example.com/b\n\n\
         [3] Title: Meta title C\n\
         URL: https://example.com/c\n\
         Description: Meta desc C\n\
         Content:\n\
         # Page C\n\n\
         Body text."
    );
    assert!(
        !out.contains("not for the model"),
        "warnings are logged, not shown"
    );
}

#[tokio::test]
async fn search_sends_every_option_and_truncates_content() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v2/search")
        .match_body(Matcher::Json(tagged(json!({
            "query": "rust news",
            "limit": 2,
            "tbs": "qdr:w",
            "location": "Germany",
            "country": "DE",
            "includeDomains": ["rust-lang.org", "docs.rs"],
            "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": true}
        }))))
        .with_header("content-type", "application/json")
        .with_body(
            json!({"success": true, "data": {"web": [{
                "url": "https://rust-lang.org/news",
                "title": "News",
                "markdown": "héllo world, a long page"
            }]}})
            .to_string(),
        )
        .create_async()
        .await;

    let search = search_tool(&server)
        .with_limit(2)
        .unwrap()
        .with_tbs("qdr:w")
        .with_location("Germany")
        .with_country("DE")
        .with_include_domains(["rust-lang.org", "docs.rs"])
        .with_scrape_results(true)
        .with_max_chars(7);
    let out = search.run(json!("rust news")).await.unwrap();
    mock.assert_async().await;
    assert_eq!(
        out,
        "[1] Title: News\nURL: https://rust-lang.org/news\nContent:\nhéllo w [truncated]"
    );
}

#[tokio::test]
async fn search_sends_exclude_domains() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v2/search")
        .match_body(Matcher::Json(tagged(json!({
            "query": "q",
            "limit": 5,
            "excludeDomains": ["example.com"]
        }))))
        .with_header("content-type", "application/json")
        .with_body(empty_search_body())
        .create_async()
        .await;
    let out = search_tool(&server)
        .with_exclude_domains(["example.com"])
        .run(json!("q"))
        .await
        .unwrap();
    mock.assert_async().await;
    assert_eq!(out, "No results found for query: q");
}

#[tokio::test]
async fn search_returns_typed_results() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v2/search")
        .with_header("content-type", "application/json")
        .with_body(search_body())
        .create_async()
        .await;

    let results = search_tool(&server).search("rust").await.unwrap();
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].title.as_deref(), Some("Announcing Rust 1.85"));
    assert_eq!(results[0].markdown, None);
    assert_eq!(results[1].description.as_deref(), Some(""));
    let metadata = results[2].metadata.as_ref().unwrap();
    assert_eq!(
        metadata.source_url.as_deref(),
        Some("https://example.com/c")
    );
    assert_eq!(metadata.status_code, Some(200));
    assert_eq!(results[2].title, None);
}

#[tokio::test]
async fn search_accepts_tool_call_and_input_objects() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v2/search")
        .match_body(Matcher::PartialJson(json!({"query": "q1"})))
        .with_header("content-type", "application/json")
        .with_body(empty_search_body())
        .expect(3)
        .create_async()
        .await;

    let search = search_tool(&server);
    for raw in [r#"{"query": "q1"}"#, r#"{"input": "q1"}"#, "q1"] {
        let out = search.call(raw).await.unwrap();
        assert_eq!(out, "No results found for query: q1", "{raw}");
    }
    mock.assert_async().await;
}

fn scrape_body() -> String {
    json!({
        "success": true,
        "data": {
            "markdown": "  # Docs\n\nHello.  ",
            "metadata": {
                "title": "Docs",
                "sourceURL": "https://example.com/docs",
                "url": "https://example.com/docs/",
                "statusCode": 200,
                "language": "en"
            },
            "warning": "served from cache"
        }
    })
    .to_string()
}

#[tokio::test]
async fn scrape_sends_defaults_with_bearer_key() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v2/scrape")
        .match_header("authorization", "Bearer test-key")
        .match_header("content-type", "application/json")
        .match_body(Matcher::Json(tagged(json!({
            "url": "https://example.com/docs",
            "formats": ["markdown"],
            "onlyMainContent": true
        }))))
        .with_header("content-type", "application/json")
        .with_body(scrape_body())
        .create_async()
        .await;

    let out = scrape_tool(&server)
        .run(json!(" https://example.com/docs "))
        .await
        .unwrap();
    mock.assert_async().await;
    assert_eq!(
        out,
        "Title: Docs\nURL: https://example.com/docs/\n\n# Docs\n\nHello."
    );
}

#[tokio::test]
async fn scrape_sends_every_option_and_shows_page_errors() {
    let mut server = mockito::Server::new_async().await;
    let mock = server
        .mock("POST", "/v2/scrape")
        .match_body(Matcher::Json(tagged(json!({
            "url": "https://example.com/missing",
            "formats": ["markdown"],
            "onlyMainContent": false,
            "includeTags": ["article"],
            "excludeTags": ["nav", ".ads"],
            "waitFor": 1500,
            "maxAge": 0
        }))))
        .with_header("content-type", "application/json")
        .with_body(
            json!({"success": true, "data": {
                "markdown": "Not found here",
                "metadata": {
                    "sourceURL": "https://example.com/missing",
                    "statusCode": 404,
                    "error": "Not Found"
                }
            }})
            .to_string(),
        )
        .create_async()
        .await;

    let scrape = scrape_tool(&server)
        .with_only_main_content(false)
        .with_include_tags(["article"])
        .with_exclude_tags(["nav", ".ads"])
        .with_wait_for(Duration::from_millis(1500))
        .with_max_age(Duration::ZERO)
        .with_max_chars(5);
    let out = scrape
        .run(json!({"url": "https://example.com/missing"}))
        .await
        .unwrap();
    mock.assert_async().await;
    assert_eq!(
        out,
        "URL: https://example.com/missing\nStatus: 404\nError: Not Found\n\nNot f [truncated]"
    );
}

#[tokio::test]
async fn scrape_without_content_says_so() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v2/scrape")
        .with_header("content-type", "application/json")
        .with_body(json!({"success": true, "data": {"markdown": "  "}}).to_string())
        .create_async()
        .await;
    let out = scrape_tool(&server)
        .run(json!("https://example.com/x"))
        .await
        .unwrap();
    assert_eq!(
        out,
        "URL: https://example.com/x\n\nNo content found on this page."
    );
}

#[tokio::test]
async fn scrape_returns_typed_document() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v2/scrape")
        .with_header("content-type", "application/json")
        .with_body(scrape_body())
        .create_async()
        .await;
    let document = scrape_tool(&server)
        .scrape("https://example.com/docs")
        .await
        .unwrap();
    assert_eq!(document.markdown.as_deref(), Some("  # Docs\n\nHello.  "));
    assert_eq!(document.metadata.title.as_deref(), Some("Docs"));
    assert_eq!(
        document.metadata.url.as_deref(),
        Some("https://example.com/docs/")
    );
}

#[tokio::test]
async fn api_errors_carry_status_and_body() {
    let mut server = mockito::Server::new_async().await;
    let _search = server
        .mock("POST", "/v2/search")
        .with_status(402)
        .with_body(format!(
            "{{\"success\": false, \"error\": \"Payment required\", \"code\": \"insufficient_credits\"}}{}",
            "x".repeat(2000)
        ))
        .create_async()
        .await;
    let _scrape = server
        .mock("POST", "/v2/scrape")
        .with_status(429)
        .with_body(r#"{"success": false, "error": "Rate limit exceeded"}"#)
        .create_async()
        .await;

    let err = search_tool(&server)
        .run(json!("q"))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("402"), "{err}");
    assert!(err.contains("Payment required"), "{err}");
    assert!(err.len() < 600, "error body is cut: {} chars", err.len());

    let err = scrape_tool(&server)
        .run(json!("https://example.com"))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("429"), "{err}");
    assert!(err.contains("Rate limit exceeded"), "{err}");
}

#[tokio::test]
async fn unsuccessful_or_malformed_responses_are_errors() {
    let mut server = mockito::Server::new_async().await;
    let _search = server
        .mock("POST", "/v2/search")
        .with_header("content-type", "application/json")
        .with_body(r#"{"success": false, "error": "Search failed upstream"}"#)
        .create_async()
        .await;
    let _scrape = server
        .mock("POST", "/v2/scrape")
        .with_header("content-type", "application/json")
        .with_body("<html>not json</html>")
        .create_async()
        .await;

    let err = search_tool(&server)
        .run(json!("q"))
        .await
        .unwrap_err()
        .to_string();
    assert_eq!(err, "Firecrawl request failed: Search failed upstream");
    assert!(scrape_tool(&server)
        .run(json!("https://example.com"))
        .await
        .is_err());
}

#[tokio::test]
async fn times_out() {
    let mut server = mockito::Server::new_async().await;
    let _mock = server
        .mock("POST", "/v2/scrape")
        .with_chunked_body(|w| {
            std::thread::sleep(Duration::from_millis(1500));
            w.write_all(b"{}")
        })
        .create_async()
        .await;
    let scrape = scrape_tool(&server).with_timeout(Duration::from_millis(200));
    let started = std::time::Instant::now();
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        scrape.run(json!("https://example.com")),
    )
    .await
    .expect("client timeout should fire first");
    assert!(result.is_err());
    assert!(started.elapsed() < Duration::from_millis(1400));
}

#[tokio::test]
async fn rejects_bad_input_before_any_request() {
    // Nothing listens on port 1; these must fail before sending.
    let search = FirecrawlSearch::new()
        .with_api_key("k")
        .with_base_url("http://127.0.0.1:1");
    for input in [
        json!(""),
        json!("  "),
        json!(42),
        json!({"q": "x"}),
        json!({"query": 1}),
    ] {
        let err = search.run(input.clone()).await.unwrap_err().to_string();
        assert!(
            err.contains("empty") || err.contains("expected a string"),
            "{input}: {err}"
        );
    }
    let err = search
        .clone()
        .with_include_domains(["a.com"])
        .with_exclude_domains(["b.com"])
        .run(json!("q"))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("cannot both be set"), "{err}");

    let scrape = FirecrawlScrape::new()
        .with_api_key("k")
        .with_base_url("http://127.0.0.1:1");
    for input in [
        json!(""),
        json!("not a url"),
        json!("example.com"),
        json!("ftp://example.com/file"),
        json!("file:///etc/passwd"),
        json!(42),
        json!({"url": 1}),
    ] {
        let err = scrape.run(input.clone()).await.unwrap_err().to_string();
        assert!(
            err.contains("URL") || err.contains("expected a string"),
            "{input}: {err}"
        );
    }

    let err = FirecrawlSearch::new()
        .with_api_key("")
        .with_base_url("http://127.0.0.1:1")
        .run(json!("q"))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("FIRECRAWL_API_KEY"), "{err}");
    let err = FirecrawlScrape::new()
        .with_api_key("")
        .with_base_url("http://127.0.0.1:1")
        .run(json!("https://example.com"))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("FIRECRAWL_API_KEY"), "{err}");
}

#[test]
fn option_validation() {
    let s = FirecrawlSearch::new;
    assert!(s().with_limit(0).is_err());
    assert!(s().with_limit(101).is_err());
    assert_eq!(s().with_limit(1).unwrap().limit, 1);
    assert_eq!(s().with_limit(100).unwrap().limit, 100);
    assert_eq!(s().with_base_url("http://x/").client.base_url, "http://x");
    assert_eq!(
        FirecrawlScrape::new()
            .with_base_url("http://x//")
            .client
            .base_url,
        "http://x"
    );
    assert_eq!(truncate("héllo", Some(5)), "héllo");
    assert_eq!(truncate("héllo", Some(2)), "hé [truncated]");
    assert_eq!(truncate("héllo", None), "héllo");
}

#[test]
fn tool_metadata() {
    let search = FirecrawlSearch::new();
    assert_eq!(search.name(), "firecrawl-search");
    assert_eq!(search.parameters()["required"], json!(["query"]));
    assert!(search.description().contains("search query"));
    let scrape = FirecrawlScrape::new();
    assert_eq!(scrape.name(), "firecrawl-scrape");
    assert_eq!(scrape.parameters()["required"], json!(["url"]));
    assert!(scrape.description().contains("URL"));
}

#[tokio::test]
async fn agent_searches_then_reads_a_page() {
    use crate::{
        agent::{AgentExecutor, OpenAiToolAgentBuilder},
        chain::Chain,
        prompt_args,
        schemas::{FunctionCallResponse, FunctionDetail, MessageType},
        test_utils::FakeLLM,
    };
    use std::sync::Arc;

    let mut server = mockito::Server::new_async().await;
    let search_mock = server
        .mock("POST", "/v2/search")
        .match_body(Matcher::PartialJson(json!({"query": "rust 1.85"})))
        .with_header("content-type", "application/json")
        .with_body(search_body())
        .create_async()
        .await;
    let scrape_mock = server
        .mock("POST", "/v2/scrape")
        .match_body(Matcher::PartialJson(
            json!({"url": "https://blog.rust-lang.org/a"}),
        ))
        .with_header("content-type", "application/json")
        .with_body(
            json!({"success": true, "data": {
                "markdown": "Rust 1.85 stabilizes async closures.",
                "metadata": {"title": "Announcing Rust 1.85", "statusCode": 200}
            }})
            .to_string(),
        )
        .create_async()
        .await;

    let call = |id: &str, name: &str, arguments: Value| FunctionCallResponse {
        id: id.into(),
        type_field: "function".into(),
        function: FunctionDetail {
            name: name.into(),
            arguments: arguments.to_string(),
        },
    };
    let llm = FakeLLM::new([
        serde_json::to_string(&vec![call(
            "call_1",
            "firecrawl-search",
            json!({"query": "rust 1.85"}),
        )])
        .unwrap(),
        serde_json::to_string(&vec![call(
            "call_2",
            "firecrawl-scrape",
            json!({"url": "https://blog.rust-lang.org/a"}),
        )])
        .unwrap(),
        "It stabilized async closures.".to_string(),
    ]);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[
            Arc::new(search_tool(&server)),
            Arc::new(scrape_tool(&server)),
        ])
        .build(llm.clone())
        .unwrap();
    let out = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "what did rust 1.85 ship?"})
        .await
        .unwrap();
    assert_eq!(out, "It stabilized async closures.");
    search_mock.assert_async().await;
    scrape_mock.assert_async().await;

    let observations: Vec<String> = llm.calls()[2]
        .iter()
        .filter(|m| m.message_type == MessageType::ToolMessage)
        .map(|m| m.content.clone())
        .collect();
    assert_eq!(observations.len(), 2, "both results go back to the model");
    assert!(observations[0].starts_with("[1] Title: Announcing Rust 1.85\n"));
    assert!(observations[1].ends_with("Rust 1.85 stabilizes async closures."));
}

/// `FIRECRAWL_API_KEY=... cargo test --lib firecrawl_live -- --ignored`
#[tokio::test]
#[ignore]
async fn firecrawl_live() {
    if std::env::var("FIRECRAWL_API_KEY").is_err() {
        eprintln!("FIRECRAWL_API_KEY not set, skipping");
        return;
    }
    let results = FirecrawlSearch::new()
        .with_limit(3)
        .unwrap()
        .search("Rust programming language")
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().all(|r| r.url.starts_with("http")));

    let out = FirecrawlScrape::new()
        .with_max_chars(2_000)
        .run(json!("https://www.rust-lang.org"))
        .await
        .unwrap();
    assert!(out.contains("Rust"), "{out}");
}
