use std::time::Duration;

use mockito::Matcher;
use serde_json::json;

use super::*;
use crate::{prompt_args, test_utils::FakeLLM};

const DOCS: &str = "GET /v1/weather?city=<name> returns {\"temp_c\": number}";

fn prompts(llm: &FakeLLM) -> Vec<String> {
    llm.calls().iter().map(|c| c[0].content.clone()).collect()
}

fn chain(llm: &FakeLLM, server: &mockito::ServerGuard) -> APIChain {
    APIChain::new(llm.clone(), DOCS, [format!("{}/v1", server.url())]).unwrap()
}

fn request_json(url: &str) -> String {
    format!(r#"{{"method": "GET", "url": "{url}", "body": null}}"#)
}

fn urls(list: &[&str]) -> Vec<Url> {
    list.iter().map(|u| parse_allowed_url(u).unwrap()).collect()
}

#[test]
fn allowed_urls_match_scheme_host_port_and_path_prefix() {
    let allowed = urls(&["https://api.example.com/v1", "http://localhost:8080/"]);
    let ok = [
        "https://api.example.com/v1",
        "https://api.example.com/v1/",
        "https://api.example.com/v1/weather?city=Lima",
        "https://api.example.com:443/v1/x",
        "https://API.example.com/v1/x",
        "http://localhost:8080/anything",
        "http://localhost:8080",
    ];
    for url in ok {
        assert!(is_allowed(&allowed, &Url::parse(url).unwrap()), "{url}");
    }
    let rejected = [
        // Other scheme, host, port, or path.
        "http://api.example.com/v1/x",
        "https://api.example.com.evil.com/v1/x",
        "https://evil.com/v1/x",
        "https://sub.api.example.com/v1/x",
        "https://api.example.com:8443/v1/x",
        "https://api.example.com/v1x",
        "https://api.example.com/v2/x",
        "https://api.example.com/",
        "http://localhost:8081/",
        "https://localhost:8080/",
        "http://127.0.0.1:8080/",
        "file:///etc/passwd",
        // Path tricks: `Url` resolves dot segments, also encoded ones.
        "https://api.example.com/v1/../admin",
        "https://api.example.com/v1/%2e%2e/admin",
        "https://api.example.com/v1/./../v2",
        // Credentials in the URL.
        "https://user:pw@api.example.com/v1/x",
        "https://api.example.com@evil.com/v1/x",
    ];
    for url in rejected {
        assert!(!is_allowed(&allowed, &Url::parse(url).unwrap()), "{url}");
    }
    // An allowed path ending in `/` still allows only what is under it.
    let allowed = urls(&["https://api.example.com/v1/"]);
    assert!(is_allowed(
        &allowed,
        &Url::parse("https://api.example.com/v1/a").unwrap()
    ));
    assert!(!is_allowed(
        &allowed,
        &Url::parse("https://api.example.com/v10").unwrap()
    ));
}

#[test]
fn constructor_validates_allowed_urls_and_methods() {
    let llm = FakeLLM::new(Vec::<String>::new());
    let err = APIChain::new(llm.clone(), DOCS, Vec::<String>::new())
        .err()
        .unwrap();
    assert!(
        err.to_string().contains("at least one allowed URL"),
        "{err}"
    );
    for bad in [
        "not a url",
        "ftp://example.com/",
        "file:///tmp/x",
        "data:text/plain,hi",
    ] {
        let err = APIChain::new(llm.clone(), DOCS, [bad]).err().unwrap();
        assert!(err.to_string().contains("allowed URL"), "{bad}: {err}");
    }
    let chain = APIChain::new(llm, DOCS, ["https://api.example.com/"]).unwrap();
    assert_eq!(chain.get_input_keys(), vec!["input"]);
    assert_eq!(
        chain.get_output_keys(),
        vec!["answer", "api_request", "api_response", DEFAULT_RESULT_KEY]
    );
    let err = chain.with_allowed_methods(["GET", "PO ST"]).err().unwrap();
    assert!(err.to_string().contains("invalid HTTP method"), "{err}");
}

#[test]
fn parse_request_formats() {
    let llm = FakeLLM::new(Vec::<String>::new());
    let chain = APIChain::new(llm, DOCS, ["https://api.example.com/v1"])
        .unwrap()
        .with_allowed_methods(["get", "POST"])
        .unwrap();
    let cases = [
        (
            r#"{"method": "GET", "url": "https://api.example.com/v1/w?city=Lima", "body": null}"#,
            Method::GET,
            "https://api.example.com/v1/w?city=Lima",
            None,
        ),
        // Fenced, with reasoning and text around it; method missing or lowercase.
        (
            "<think>{\"url\": \"https://evil.com/\"}</think>Sure:\n```json\n{\"url\": \" https://api.example.com/v1/a \"}\n```",
            Method::GET,
            "https://api.example.com/v1/a",
            None,
        ),
        (
            r#"Here: {"method": "post", "url": "https://api.example.com/v1/items", "body": {"name": "x", "n": 2}}"#,
            Method::POST,
            "https://api.example.com/v1/items",
            Some(json!({"name": "x", "n": 2})),
        ),
    ];
    for (output, method, url, body) in cases {
        let request = chain.parse_request(output).unwrap();
        assert_eq!(request.method, method, "{output}");
        assert_eq!(request.url.as_str(), url, "{output}");
        assert_eq!(request.body, body, "{output}");
    }

    let errors = [
        ("no json here", "no JSON found"),
        (r#"{"method": "GET"}"#, "no `url`"),
        (r#"{"url": 5}"#, "no `url`"),
        (r#"{"url": "not a url"}"#, "invalid url"),
        (
            r#"{"method": 1, "url": "https://api.example.com/v1/"}"#,
            "`method` must be a string",
        ),
        (
            r#"{"url": "https://evil.com/v1/"}"#,
            "https://evil.com/v1/ is not an allowed URL",
        ),
        (
            r#"{"url": "http://169.254.169.254/latest/meta-data/"}"#,
            "is not an allowed URL",
        ),
        (
            r#"{"method": "DELETE", "url": "https://api.example.com/v1/x"}"#,
            "method DELETE is not allowed",
        ),
    ];
    for (output, expected) in errors {
        let err = chain.parse_request(output).unwrap_err().to_string();
        assert!(err.contains(expected), "{output}: {err}");
    }
}

#[tokio::test]
async fn asks_calls_and_answers() {
    let mut server = mockito::Server::new_async().await;
    let api = server
        .mock("GET", "/v1/weather")
        .match_query(Matcher::UrlEncoded("city".into(), "Lima".into()))
        .match_header("x-api-key", "secret")
        .with_status(200)
        .with_header("content-type", "application/json")
        .with_body(r#"{"temp_c": 19.5}"#)
        .expect(1)
        .create_async()
        .await;
    let url = format!("{}/v1/weather?city=Lima", server.url());
    let llm = FakeLLM::new([request_json(&url), "It's 19.5 °C in Lima.".to_string()]);
    let chain = chain(&llm, &server).with_header("x-api-key", "secret");

    let out = chain
        .execute(prompt_args! {"input" => "How warm is Lima?"})
        .await
        .unwrap();
    api.assert_async().await;
    assert_eq!(out["answer"], json!("It's 19.5 °C in Lima."));
    assert_eq!(out[API_CHAIN_REQUEST_KEY], json!(format!("GET {url}")));
    assert_eq!(out[API_CHAIN_RESPONSE_KEY], json!(r#"{"temp_c": 19.5}"#));
    let result: GenerateResult = serde_json::from_value(out[DEFAULT_RESULT_KEY].clone()).unwrap();
    assert_eq!(result.generation, "It's 19.5 °C in Lima.");
    // Both LLM calls' tokens (FakeLLM reports 1 + 1 each).
    assert_eq!(result.tokens.unwrap().total_tokens, 4);

    let prompts = prompts(&llm);
    assert_eq!(prompts.len(), 2);
    assert!(prompts[0].contains(DOCS), "{}", prompts[0]);
    assert!(
        prompts[0].contains("Question: How warm is Lima?"),
        "{}",
        prompts[0]
    );
    assert!(
        !prompts[0].contains("{{"),
        "unrendered variable: {}",
        prompts[0]
    );
    assert!(
        !prompts[0].contains("secret"),
        "headers are not shown to the model"
    );
    assert!(
        prompts[1].contains("Question: How warm is Lima?"),
        "{}",
        prompts[1]
    );
    assert!(prompts[1].contains(&format!("GET {url}")), "{}", prompts[1]);
    assert!(prompts[1].contains(r#"{"temp_c": 19.5}"#), "{}", prompts[1]);
    assert!(
        !prompts[1].contains("{{"),
        "unrendered variable: {}",
        prompts[1]
    );
}

#[tokio::test]
async fn invoke_with_custom_keys() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/v1/x")
        .with_body("pong")
        .create_async()
        .await;
    let llm = FakeLLM::new([
        request_json(&format!("{}/v1/x", server.url())),
        "pong!".to_string(),
    ]);
    let chain = chain(&llm, &server)
        .with_input_key("question")
        .with_output_key("reply");
    assert_eq!(chain.get_input_keys(), vec!["question"]);
    let err = chain
        .call(prompt_args! {"input" => "ping"})
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ChainError::MissingInputVariable(k) if k == "question"),
        "{err:?}"
    );
    assert!(llm.calls().is_empty());

    let out = chain
        .execute(prompt_args! {"question" => "ping"})
        .await
        .unwrap();
    assert_eq!(out["reply"], json!("pong!"));
    assert!(!out.contains_key("answer"));
    assert!(prompts(&llm)[0].contains("Question: ping"));
}

#[tokio::test]
async fn disallowed_request_is_never_sent() {
    let mut server = mockito::Server::new_async().await;
    let outside = server
        .mock("GET", Matcher::Any)
        .expect(0)
        .create_async()
        .await;
    let llm = FakeLLM::new([request_json(&format!("{}/admin", server.url()))]);
    let err = chain(&llm, &server)
        .invoke(prompt_args! {"input" => "x"})
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("/admin is not an allowed URL"), "{err}");
    outside.assert_async().await;
    assert_eq!(
        llm.calls().len(),
        1,
        "no answer call after a rejected request"
    );

    let llm = FakeLLM::new([format!(
        r#"{{"method": "POST", "url": "{}/v1/x", "body": {{"a": 1}}}}"#,
        server.url()
    )]);
    let err = chain(&llm, &server)
        .invoke(prompt_args! {"input" => "x"})
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("method POST is not allowed"), "{err}");
    outside.assert_async().await;
}

#[tokio::test]
async fn sends_a_json_body_for_allowed_methods() {
    let mut server = mockito::Server::new_async().await;
    let api = server
        .mock("POST", "/v1/items")
        .match_header("content-type", "application/json")
        .match_body(Matcher::Json(json!({"name": "pen", "qty": 2})))
        .with_status(201)
        .with_body(r#"{"id": 7}"#)
        .create_async()
        .await;
    let llm = FakeLLM::new([
        format!(
            r#"{{"method": "POST", "url": "{}/v1/items", "body": {{"name": "pen", "qty": 2}}}}"#,
            server.url()
        ),
        "Created item 7.".to_string(),
    ]);
    let post_chain = chain(&llm, &server)
        .with_allowed_methods(["GET", "POST"])
        .unwrap();
    assert_eq!(
        post_chain
            .invoke(prompt_args! {"input" => "add 2 pens"})
            .await
            .unwrap(),
        "Created item 7."
    );
    api.assert_async().await;

    // A GET never sends the body the model wrote.
    let get = server
        .mock("GET", "/v1/items")
        .match_body(Matcher::Exact(String::new()))
        .with_body("[]")
        .create_async()
        .await;
    let llm = FakeLLM::new([
        format!(
            r#"{{"method": "GET", "url": "{}/v1/items", "body": {{"x": 1}}}}"#,
            server.url()
        ),
        "none".to_string(),
    ]);
    chain(&llm, &server)
        .invoke(prompt_args! {"input" => "list"})
        .await
        .unwrap();
    get.assert_async().await;
}

#[tokio::test]
async fn error_status_is_an_error_with_the_body() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/v1/x")
        .with_status(404)
        .with_body("city not found")
        .create_async()
        .await;
    let llm = FakeLLM::new([request_json(&format!("{}/v1/x", server.url()))]);
    let err = chain(&llm, &server)
        .invoke(prompt_args! {"input" => "x"})
        .await
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("status 404") && err.contains("city not found"),
        "{err}"
    );
    assert!(err.contains(&format!("GET {}/v1/x", server.url())), "{err}");
    assert_eq!(llm.calls().len(), 1);
}

#[tokio::test]
async fn long_responses_are_truncated() {
    let mut server = mockito::Server::new_async().await;
    server
        .mock("GET", "/v1/big")
        .with_chunked_body(|w| {
            for _ in 0..5 {
                w.write_all(&[b'a'; 1000])?;
            }
            Ok(())
        })
        .create_async()
        .await;
    server
        .mock("GET", "/v1/small")
        .with_body("0123456789")
        .create_async()
        .await;
    let url = server.url();
    let llm = FakeLLM::new([
        request_json(&format!("{url}/v1/big")),
        "ok".into(),
        request_json(&format!("{url}/v1/small")),
        "ok".into(),
    ]);
    let chain = chain(&llm, &server).with_max_response_bytes(2500);
    let out = chain.execute(prompt_args! {"input" => "x"}).await.unwrap();
    assert_eq!(
        out[API_CHAIN_RESPONSE_KEY],
        json!(format!("{}{TRUNCATED}", "a".repeat(2500)))
    );
    // Exactly at the limit is not truncated.
    let chain = chain.with_max_response_bytes(10);
    let out = chain.execute(prompt_args! {"input" => "x"}).await.unwrap();
    assert_eq!(out[API_CHAIN_RESPONSE_KEY], json!("0123456789"));
}

#[tokio::test]
async fn redirects_are_followed_only_to_allowed_urls() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();
    server
        .mock("GET", "/v1/old")
        .with_status(302)
        .with_header("location", "/v1/new")
        .create_async()
        .await;
    server
        .mock("GET", "/v1/new")
        .with_body("moved here")
        .create_async()
        .await;
    server
        .mock("GET", "/v1/escape")
        .with_status(302)
        .with_header("location", "/internal")
        .create_async()
        .await;
    let internal = server
        .mock("GET", "/internal")
        .with_body("secret")
        .expect(0)
        .create_async()
        .await;
    server
        .mock("GET", "/v1/loop")
        .with_status(302)
        .with_header("location", "/v1/loop")
        .create_async()
        .await;

    let llm = FakeLLM::new([
        request_json(&format!("{url}/v1/old")),
        "ok".into(),
        request_json(&format!("{url}/v1/escape")),
        request_json(&format!("{url}/v1/loop")),
    ]);
    let chain = chain(&llm, &server);
    let out = chain.execute(prompt_args! {"input" => "x"}).await.unwrap();
    assert_eq!(out[API_CHAIN_RESPONSE_KEY], json!("moved here"));

    let err = chain
        .invoke(prompt_args! {"input" => "x"})
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("/internal is not an allowed URL"), "{err}");
    internal.assert_async().await;

    let err = chain
        .invoke(prompt_args! {"input" => "x"})
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("too many redirects"), "{err}");
}

#[tokio::test]
async fn redirect_chains_stop_after_five_hops() {
    let mut server = mockito::Server::new_async().await;
    let url = server.url();
    // /v1/hop/0 -> /v1/hop/1 -> ... -> /v1/hop/N, which answers.
    for i in 0..7 {
        server
            .mock("GET", format!("/v1/hop/{i}").as_str())
            .with_status(302)
            .with_header("location", &format!("/v1/hop/{}", i + 1))
            .create_async()
            .await;
    }
    server
        .mock("GET", "/v1/hop/7")
        .with_body("end 7")
        .create_async()
        .await;
    // Starting at hop 2: five redirects (2->7) are allowed.
    let llm = FakeLLM::new([
        request_json(&format!("{url}/v1/hop/2")),
        "ok".into(),
        request_json(&format!("{url}/v1/hop/0")),
    ]);
    let chain = chain(&llm, &server);
    let out = chain.execute(prompt_args! {"input" => "x"}).await.unwrap();
    assert_eq!(out[API_CHAIN_RESPONSE_KEY], json!("end 7"));
    // Starting at hop 0: seven redirects are too many.
    let err = chain
        .invoke(prompt_args! {"input" => "x"})
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("too many redirects"), "{err}");
}

#[tokio::test]
async fn requests_time_out() {
    // A server that accepts the connection and never answers.
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let hold = tokio::spawn(async move {
        let mut conns = Vec::new();
        loop {
            let (conn, _) = listener.accept().await.unwrap();
            conns.push(conn);
        }
    });
    let llm = FakeLLM::new([request_json(&format!("http://{addr}/v1/slow"))]);
    let chain = APIChain::new(llm, DOCS, [format!("http://{addr}/v1")])
        .unwrap()
        .with_timeout(Duration::from_millis(200));
    let err = tokio::time::timeout(
        Duration::from_secs(5),
        chain.invoke(prompt_args! {"input" => "x"}),
    )
    .await
    .expect("the request timeout did not fire")
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("timed out") || err.contains("timeout"),
        "{err}"
    );
    hold.abort();
}

#[tokio::test]
async fn llm_errors_pass_through() {
    let server = mockito::Server::new_async().await;
    let llm = FakeLLM::new(Vec::<String>::new());
    let err = chain(&llm, &server)
        .invoke(prompt_args! {"input" => "x"})
        .await
        .unwrap_err();
    assert!(matches!(err, ChainError::LLMError(_)), "{err:?}");
}

/// `API_CHAIN_OLLAMA_MODEL=qwen3:14b cargo test --lib api_chain_live -- --ignored --nocapture`
///
/// A real model writes the request against a local mock API and answers from it.
#[tokio::test]
#[ignore]
async fn api_chain_live_ollama() {
    use crate::llm::{openai::OpenAI, OpenAIConfig};

    let model = std::env::var("API_CHAIN_OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:14b".into());
    let llm = OpenAI::default()
        .with_config(
            OpenAIConfig::default()
                .with_api_base("http://localhost:11434/v1")
                .with_api_key("ollama"),
        )
        .with_model(model);
    let mut server = mockito::Server::new_async().await;
    let api = server
        .mock("GET", "/v1/weather")
        .match_query(Matcher::UrlEncoded("city".into(), "Lima".into()))
        .with_header("content-type", "application/json")
        .with_body(r#"{"city": "Lima", "temp_c": 19.5, "sky": "overcast"}"#)
        .expect_at_least(1)
        .create_async()
        .await;
    let docs = format!(
        "Weather API. Base URL: {}/v1\n\nGET /weather?city=<city name>\nReturns JSON: {{\"city\": string, \"temp_c\": number, \"sky\": string}}",
        server.url()
    );
    let chain = APIChain::new(llm, docs, [format!("{}/v1", server.url())]).unwrap();
    let out = chain
        .execute(prompt_args! {"input" => "What's the temperature in Lima right now?"})
        .await
        .unwrap();
    println!(
        "request: {}\nanswer: {}",
        out[API_CHAIN_REQUEST_KEY], out["answer"]
    );
    api.assert_async().await;
    assert!(
        out["answer"].as_str().unwrap().contains("19.5"),
        "{}",
        out["answer"]
    );
}
