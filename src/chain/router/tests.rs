use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

use super::{chain::parse_destination, *};
use crate::{
    chain::{Chain, ChainError, LLMChainBuilder},
    language_models::{GenerateResult, LLMEvent, TokenUsage},
    prompt::{HumanMessagePromptTemplate, PromptArgs},
    prompt_args,
    schemas::MessageType,
    template_fstring,
    test_utils::FakeLLM,
};

/// A destination that records its input and answers `"{name}: {input}"`.
#[derive(Clone)]
struct Spy {
    name: &'static str,
    seen: Arc<Mutex<Vec<PromptArgs>>>,
}

impl Spy {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            seen: Default::default(),
        }
    }
    fn calls(&self) -> Vec<PromptArgs> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl Chain for Spy {
    async fn call(&self, input: PromptArgs) -> Result<GenerateResult, ChainError> {
        let text = format!("{}: {}", self.name, input["input"].as_str().unwrap_or(""));
        self.seen.lock().unwrap().push(input);
        Ok(GenerateResult {
            generation: text,
            tokens: Some(TokenUsage::new(10, 5)),
            ..Default::default()
        })
    }
}

fn router(llm: FakeLLM, physics: &Spy, math: &Spy, default: Option<&Spy>) -> RouterChain {
    let mut builder = RouterChainBuilder::new()
        .llm(llm)
        .destination("physics", "Questions about physics", physics.clone())
        .destination(" math ", " Math problems ", math.clone());
    if let Some(default) = default {
        builder = builder.default_chain(default.clone());
    }
    builder.build().unwrap()
}

#[test]
fn parses_the_destination() {
    assert_eq!(parse_destination(r#"{"destination": "math"}"#), "math");
    assert_eq!(
        parse_destination("```json\n{\"destination\": \" physics \"}\n```"),
        "physics"
    );
    assert_eq!(
        parse_destination(
            "Sure! {\"destination\": \"DEFAULT\", \"why\": \"none fit\"} hope it helps"
        ),
        "DEFAULT"
    );
    assert_eq!(parse_destination("  \"math\". "), "math");
    assert_eq!(parse_destination("`physics`"), "physics");
    assert_eq!(
        parse_destination(r#"{"destination": "math", "why": {"confidence": 0.9}}"#),
        "math",
        "nested objects"
    );
    // Not the JSON we asked for: fall back to the raw answer.
    assert_eq!(
        parse_destination(r#"{"route": "math"}"#),
        r#"{"route": "math"}"#
    );
    assert_eq!(parse_destination("} math {"), "} math {");
    assert_eq!(
        parse_destination(r#"{"destination": 3}"#),
        r#"{"destination": 3}"#
    );
}

#[tokio::test]
async fn routes_to_the_chosen_destination_with_the_full_input() {
    let (physics, math) = (Spy::new("physics"), Spy::new("math"));
    let llm = FakeLLM::new([r#"{"destination": "MATH"}"#]);
    let chain = router(llm.clone(), &physics, &math, None);

    let result = chain
        .call(prompt_args! { "input" => "2 + 2?", "lang" => "es" })
        .await
        .unwrap();
    assert_eq!(result.generation, "math: 2 + 2?");
    // Destination tokens (10/5) + routing call (1/1).
    assert_eq!(result.tokens, Some(TokenUsage::new(11, 6)));
    assert!(physics.calls().is_empty());
    let seen = math.calls();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0]["lang"], "es", "every input is forwarded");
    assert_eq!(seen[0]["input"], "2 + 2?");

    // The router prompt lists every destination, trimmed, and the input goes as is.
    let calls = llm.calls();
    assert_eq!(calls.len(), 1);
    let [system, human] = &calls[0][..] else {
        panic!("{:?}", calls[0])
    };
    assert_eq!(system.message_type, MessageType::SystemMessage);
    assert!(
        system
            .content
            .contains("physics: Questions about physics\nmath: Math problems\n"),
        "{}",
        system.content
    );
    assert!(system.content.contains("DEFAULT"), "{}", system.content);
    assert_eq!(human.message_type, MessageType::HumanMessage);
    assert_eq!(human.content, "2 + 2?");
}

#[tokio::test]
async fn default_and_unknown_answers_use_the_default_chain() {
    for answer in [r#"{"destination": "DEFAULT"}"#, "default", "chemistry"] {
        let (physics, math, general) = (Spy::new("physics"), Spy::new("math"), Spy::new("general"));
        let chain = router(FakeLLM::new([answer]), &physics, &math, Some(&general));
        let result = chain
            .call(prompt_args! { "input" => "hello" })
            .await
            .unwrap();
        assert_eq!(result.generation, "general: hello", "{answer}");
        assert_eq!(general.calls().len(), 1);
        assert!(physics.calls().is_empty() && math.calls().is_empty());
    }
}

#[tokio::test]
async fn without_a_default_unroutable_inputs_fail() {
    let (physics, math) = (Spy::new("physics"), Spy::new("math"));
    let chain = router(FakeLLM::new(["chemistry"]), &physics, &math, None);
    let err = chain
        .call(prompt_args! { "input" => "x" })
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("\"chemistry\""), "{err}");
    assert!(err.contains("[\"physics\", \"math\"]"), "{err}");

    let chain = router(FakeLLM::new(["DEFAULT", "DEFAULT"]), &physics, &math, None);
    // Answering DEFAULT is a valid decision even without a default chain...
    assert_eq!(
        chain
            .route(&prompt_args! { "input" => "x" })
            .await
            .unwrap()
            .destination,
        None
    );
    // ...but there's nothing to run.
    let err = chain
        .call(prompt_args! { "input" => "x" })
        .await
        .unwrap_err()
        .to_string();
    assert_eq!(
        err,
        "Error: no destination fits and there is no default chain"
    );
    assert!(physics.calls().is_empty() && math.calls().is_empty());
}

#[tokio::test]
async fn route_only_decides() {
    let (physics, math) = (Spy::new("physics"), Spy::new("math"));
    let chain = router(
        FakeLLM::new([r#"{"destination": "physics"}"#, "DEFAULT"]),
        &physics,
        &math,
        Some(&Spy::new("general")),
    );
    let input = prompt_args! { "input" => "gravity?" };
    assert_eq!(
        chain.route(&input).await.unwrap(),
        Route {
            destination: Some("physics".into()),
            tokens: Some(TokenUsage::new(1, 1)),
        }
    );
    assert_eq!(chain.route(&input).await.unwrap().destination, None);
    assert!(physics.calls().is_empty(), "route doesn't run the chain");
}

#[tokio::test]
async fn errors_are_returned() {
    let (physics, math) = (Spy::new("physics"), Spy::new("math"));
    // Missing input: fails before calling the LLM.
    let llm = FakeLLM::new(["math"]);
    let chain = router(llm.clone(), &physics, &math, None);
    let err = chain
        .call(prompt_args! { "question" => "x" })
        .await
        .unwrap_err();
    assert!(
        matches!(err, ChainError::MissingInputVariable(ref k) if k == "input"),
        "{err}"
    );
    assert!(llm.calls().is_empty());

    // LLM failure.
    let chain = router(FakeLLM::default(), &physics, &math, None);
    let err = chain
        .call(prompt_args! { "input" => "x" })
        .await
        .unwrap_err();
    assert!(matches!(err, ChainError::LLMError(_)), "{err}");

    // Destination failure.
    let chain = RouterChainBuilder::new()
        .llm(FakeLLM::new(["bad"]))
        .destination("bad", "always fails", greet_chain(FakeLLM::default()))
        .build()
        .unwrap();
    let err = chain
        .call(prompt_args! { "input" => "x", "name" => "a" })
        .await
        .unwrap_err();
    assert!(matches!(err, ChainError::LLMError(_)), "{err}");
}

#[tokio::test]
async fn custom_input_and_output_keys() {
    let llm = FakeLLM::new([r#"{"destination": "greet"}"#, "greet"]);
    let chain = RouterChainBuilder::new()
        .llm(llm.clone())
        .destination(
            "greet",
            "Greets people",
            greet_chain(FakeLLM::new(["hi a", "hi b"])),
        )
        .input_key("name")
        .output_key("answer")
        .build()
        .unwrap();
    assert_eq!(chain.get_input_keys(), ["name"]);
    assert_eq!(
        chain.get_output_keys(),
        ["answer", "generate_result", "destination"]
    );

    let output = chain.execute(prompt_args! { "name" => "a" }).await.unwrap();
    assert_eq!(output["answer"], "hi a");
    assert_eq!(output["destination"], "greet");
    assert_eq!(
        output["output"], "hi a",
        "the destination's own outputs are kept"
    );
    let result: GenerateResult = serde_json::from_value(output["generate_result"].clone()).unwrap();
    assert_eq!(result.generation, "hi a");
    assert_eq!(result.tokens, Some(TokenUsage::new(2, 2)));
    assert_eq!(llm.calls()[0][1].content, "a");

    let answer = chain.invoke(prompt_args! { "name" => "b" }).await.unwrap();
    assert_eq!(answer, "hi b");
}

#[tokio::test]
async fn execute_reports_the_default_destination() {
    let (physics, math, general) = (Spy::new("physics"), Spy::new("math"), Spy::new("general"));
    let chain = router(FakeLLM::new(["DEFAULT"]), &physics, &math, Some(&general));
    let output = chain
        .execute(prompt_args! { "input" => "hey" })
        .await
        .unwrap();
    assert_eq!(output["destination"], "DEFAULT");
    assert_eq!(output["output"], "general: hey");
    let result: GenerateResult = serde_json::from_value(output["generate_result"].clone()).unwrap();
    assert_eq!(result.tokens, Some(TokenUsage::new(11, 6)));
}

#[tokio::test]
async fn execute_without_a_result_key_uses_the_first_output_key() {
    struct Plain(&'static str);
    #[async_trait]
    impl Chain for Plain {
        async fn call(&self, _: PromptArgs) -> Result<GenerateResult, ChainError> {
            unreachable!()
        }
        async fn execute(
            &self,
            _: PromptArgs,
        ) -> Result<std::collections::HashMap<String, serde_json::Value>, ChainError> {
            Ok([(self.0.to_string(), json!("plain answer"))].into())
        }
        fn get_output_keys(&self) -> Vec<String> {
            match self.0 {
                // No declared keys: the answer is expected under `output`.
                "output" => vec![],
                key => vec![key.into()],
            }
        }
    }
    for key in ["text", "output"] {
        let chain = RouterChainBuilder::new()
            .llm(FakeLLM::new(["plain"]))
            .destination("plain", "plain", Plain(key))
            .output_key("answer")
            .build()
            .unwrap();
        let output = chain
            .execute(prompt_args! { "input" => "x" })
            .await
            .unwrap();
        assert_eq!(output["answer"], "plain answer", "{key}");
        assert_eq!(output[key], "plain answer");
        let result: GenerateResult =
            serde_json::from_value(output["generate_result"].clone()).unwrap();
        assert_eq!(result.generation, "plain answer");
        assert_eq!(result.tokens, Some(TokenUsage::new(1, 1)));
    }
}

#[tokio::test]
async fn call_stream_routes_then_streams() {
    let chain = RouterChainBuilder::new()
        .llm(FakeLLM::new(["greet"]))
        .destination(
            "greet",
            "Greets",
            greet_chain(FakeLLM::new(["hello there friend"])),
        )
        .input_key("name")
        .build()
        .unwrap();
    let events: Vec<_> = chain
        .call_stream(prompt_args! { "name" => "a" })
        .collect()
        .await;
    let events: Vec<LLMEvent> = events.into_iter().map(Result::unwrap).collect();
    let texts: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            LLMEvent::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, ["hello ", "there ", "friend"], "streamed for real");
    let Some(LLMEvent::Done(done)) = events.last() else {
        panic!("{events:?}")
    };
    assert_eq!(done.generation, "hello there friend");
    assert_eq!(done.tokens, Some(TokenUsage::new(2, 2)));

    let (physics, math) = (Spy::new("physics"), Spy::new("math"));
    let chain = router(FakeLLM::new(["chemistry"]), &physics, &math, None);
    let events: Vec<_> = chain
        .call_stream(prompt_args! { "input" => "x" })
        .collect()
        .await;
    assert_eq!(events.len(), 1);
    assert!(events[0]
        .as_ref()
        .unwrap_err()
        .to_string()
        .contains("chemistry"));
}

#[test]
fn builder_validates() {
    let err = |b: RouterChainBuilder| b.build().err().unwrap().to_string();
    let spy = Spy::new("x");
    assert!(err(RouterChainBuilder::new().destination("a", "a", spy.clone())).contains("llm"));
    assert!(err(RouterChainBuilder::new().llm(FakeLLM::default())).contains("destination"));
    for (a, b, expected) in [
        ("math", " MATH ", "duplicate"),
        ("math", "default", "invalid"),
        ("math", "  ", "invalid"),
    ] {
        let e = err(RouterChainBuilder::new()
            .llm(FakeLLM::default())
            .destination(a, "a", spy.clone())
            .destination(b, "b", spy.clone()));
        assert!(e.contains(expected), "{a}/{b}: {e}");
    }
}

#[tokio::test]
async fn call_runs_in_a_router_span() {
    let (spans, _guard) = crate::observability::testing::capture();
    let (physics, math) = (Spy::new("physics"), Spy::new("math"));
    router(FakeLLM::new(["physics"]), &physics, &math, None)
        .call(prompt_args! { "input" => "x" })
        .await
        .unwrap();
    router(
        FakeLLM::new(["DEFAULT"]),
        &physics,
        &math,
        Some(&Spy::new("general")),
    )
    .execute(prompt_args! { "input" => "x" })
    .await
    .unwrap();
    router(FakeLLM::new(["nope"]), &physics, &math, None)
        .call(prompt_args! { "input" => "x" })
        .await
        .unwrap_err();
    let spans: Vec<_> = spans
        .named("chain.call")
        .into_iter()
        .filter(|s| s.field("chain") == Some("router"))
        .collect();
    assert_eq!(spans.len(), 3);
    assert_eq!(spans[0].field("destination"), Some("physics"));
    assert_eq!(spans[1].field("destination"), Some("DEFAULT"));
    assert!(spans[2].field("error").unwrap().contains("nope"));
}

fn greet_chain(llm: FakeLLM) -> crate::chain::LLMChain {
    LLMChainBuilder::new()
        .prompt(HumanMessagePromptTemplate::new(template_fstring!(
            "Hello {name}!",
            "name"
        )))
        .llm(llm)
        .build()
        .unwrap()
}
