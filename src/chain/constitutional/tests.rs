use serde_json::json;

use super::*;
use crate::{language_models::LLMError, template_jinja2, test_utils::FakeLLM};

fn qa_chain(llm: &FakeLLM) -> LLMChain {
    LLMChainBuilder::new()
        .prompt(template_jinja2!("Q: {{question}}", "question"))
        .llm(llm.clone())
        .build()
        .unwrap()
}

fn principle(name: &str) -> ConstitutionalPrinciple {
    ConstitutionalPrinciple::new(format!("critique {name}?"), format!("revise {name}!"))
        .with_name(name)
}

fn prompts(llm: &FakeLLM) -> Vec<String> {
    llm.calls().iter().map(|c| c[0].content.clone()).collect()
}

/// The text after the last few-shot separator: the part filled with this call's inputs.
fn last_part(prompt: &str) -> &str {
    prompt.rsplit(EXAMPLE_SEPARATOR).next().unwrap()
}

#[test]
fn builtin_principles() {
    let names: Vec<_> = ConstitutionalPrinciple::builtin_names().collect();
    assert_eq!(names.len(), 49);
    let unique: std::collections::HashSet<_> = names.iter().collect();
    assert_eq!(unique.len(), names.len());

    let illegal = ConstitutionalPrinciple::get("illegal").unwrap();
    assert_eq!(illegal.name, "illegal");
    assert!(illegal
        .critique_request
        .starts_with("Discuss whether the assistant provided any advice"));
    assert!(illegal.revision_request.contains("legally questionable"));
    for name in names {
        let p = ConstitutionalPrinciple::get(name).unwrap();
        assert!(
            !p.critique_request.is_empty() && !p.revision_request.is_empty(),
            "{name}"
        );
    }
    assert_eq!(ConstitutionalPrinciple::get("nope"), None);
    assert_eq!(
        ConstitutionalPrinciple::get("harmful"),
        None,
        "exact names only"
    );
    assert_eq!(
        ConstitutionalPrinciple::new("c", "r").name,
        "Constitutional Principle"
    );
}

#[test]
fn few_shot_templates() {
    let critique = critique_template();
    assert_eq!(
        critique.matches(EXAMPLE_SEPARATOR).count(),
        5,
        "intro + 4 examples + input"
    );
    assert!(critique.starts_with(
        "Below is a conversation between a human and an AI model. If there is no material critique"
    ));
    assert!(critique.contains("Model: Sure, I can help you burn your house down."));
    assert!(critique.contains("Critique Needed.") && critique.contains("No critique needed."));
    assert!(
        !critique.contains("Revision Request:"),
        "critique examples stop at the critique"
    );
    assert!(critique.ends_with("Critique Request: {{critique_request}}\n\nCritique:"));

    let revision = revision_template();
    assert_eq!(revision.matches(EXAMPLE_SEPARATOR).count(), 5);
    assert!(revision.contains("Revision: I would strongly advise against burning your house down"));
    assert!(revision.contains("Revision: No revisions needed."));
    assert!(revision.ends_with("Revision Request: {{revision_request}}\n\nRevision:"));
}

#[test]
fn parse_critique_formats() {
    let cases = [
        (
            "It is fine. No critique needed.",
            "It is fine. No critique needed.",
        ),
        (
            "  Too rude. Critique needed.\n\n",
            "Too rude. Critique needed.",
        ),
        (
            "Too rude. Critique needed.\n\nRevision request: be nice",
            "Too rude. Critique needed.",
        ),
        ("Bad.\nRevision Request: x", "Bad."),
        (
            "<think>hmm</think>\nBad. Critique needed.",
            "Bad. Critique needed.",
        ),
        // An echoed label (seen with gemma); a later "critique:" stays.
        ("Critique: Rude. Critique needed.", "Rude. Critique needed."),
        ("critique:Rude", "Rude"),
        ("Fine. critique: none", "Fine. critique: none"),
        // Non-ASCII text before the cut doesn't shift it (the Kelvin sign lowercases to a
        // shorter `k`).
        ("Ñandú… Revision request: x", "Ñandú…"),
        ("300 \u{212A} hot. Revision request: x", "300 \u{212A} hot."),
    ];
    for (raw, want) in cases {
        assert_eq!(parse_critique(raw), want, "{raw:?}");
    }
}

#[test]
fn parse_revision_formats() {
    let cases = [
        ("Spiders have eight legs.", "Spiders have eight legs."),
        // An echoed label (seen with qwen3).
        (
            "Revision: Spiders have eight legs.",
            "Spiders have eight legs.",
        ),
        ("  REVISION:eight\n", "eight"),
        ("<think>x</think> Revision: eight", "eight"),
        ("The revision: none", "The revision: none"),
        ("Ñ", "Ñ"),
    ];
    for (raw, want) in cases {
        assert_eq!(parse_revision(raw), want, "{raw:?}");
    }
}

#[tokio::test]
async fn critiques_and_revises_in_order() {
    let llm = FakeLLM::new([
        "first answer",
        "It is rude. Critique needed.\n\nRevision request: ignored",
        "polite answer",
        "Looks good. No critique needed.",
        "Still vague. Critique needed.",
        "<think>ok</think>\nprecise answer",
    ]);
    let chain = ConstitutionalChain::new(
        qa_chain(&llm),
        llm.clone(),
        vec![principle("polite"), principle("safe"), principle("precise")],
    );
    let out = chain
        .critique_and_revise(prompt_args! {"question" => "why?"})
        .await
        .unwrap();
    assert_eq!(out.initial_output, "first answer");
    assert_eq!(out.output, "precise answer");
    assert_eq!(out.tokens.unwrap().total_tokens, 12, "usage of all 6 calls");
    assert_eq!(
        out.critiques_and_revisions,
        vec![
            CritiqueRevision {
                principle: "polite".into(),
                critique: "It is rude. Critique needed.".into(),
                revision: Some("polite answer".into()),
            },
            CritiqueRevision {
                principle: "safe".into(),
                critique: "Looks good. No critique needed.".into(),
                revision: None,
            },
            CritiqueRevision {
                principle: "precise".into(),
                critique: "Still vague. Critique needed.".into(),
                revision: Some("precise answer".into()),
            },
        ]
    );

    let p = prompts(&llm);
    assert_eq!(p.len(), 6);
    assert_eq!(p[0], "Q: why?");
    assert_eq!(
        last_part(&p[1]),
        "Human: Q: why?\n\nModel: first answer\n\nCritique Request: critique polite?\n\nCritique:"
    );
    let revise = last_part(&p[2]);
    assert!(revise.starts_with("Human: Q: why?\n\nModel: first answer\n\nCritique Request: critique polite?\n\nCritique: It is rude. Critique needed.\n\n"), "{revise}");
    assert!(
        revise.ends_with("Revision Request: revise polite!\n\nRevision:"),
        "{revise}"
    );
    // Later principles see the revised answer.
    assert!(last_part(&p[3]).contains("Model: polite answer\n"));
    assert!(
        last_part(&p[4]).contains("Model: polite answer\n\nCritique Request: critique precise?")
    );
    assert!(last_part(&p[5]).contains("Critique: Still vague. Critique needed.\n"));
}

#[tokio::test]
async fn no_revisions_needed_keeps_the_answer() {
    let llm = FakeLLM::new(["answer", "Hmm. Critique needed.", "No revisions needed."]);
    let chain = ConstitutionalChain::new(qa_chain(&llm), llm.clone(), vec![principle("p")]);
    let out = chain
        .critique_and_revise(prompt_args! {"question" => "q"})
        .await
        .unwrap();
    assert_eq!(out.output, "answer");
    assert_eq!(out.critiques_and_revisions[0].revision, None);
}

#[tokio::test]
async fn without_principles_returns_the_first_answer() {
    let llm = FakeLLM::new(["<think>x</think> just this "]);
    let chain = ConstitutionalChain::new(qa_chain(&llm), llm.clone(), vec![]);
    assert_eq!(
        chain
            .invoke(prompt_args! {"question" => "q"})
            .await
            .unwrap(),
        "just this"
    );
    assert_eq!(llm.calls().len(), 1);
}

#[tokio::test]
async fn execute_output_keys() {
    let make = |steps: bool| {
        let llm = FakeLLM::new(["a", "Bad. Critique needed.", "b"]);
        ConstitutionalChain::new(qa_chain(&llm), llm, vec![principle("p")])
            .with_output_key("answer")
            .with_return_intermediate_steps(steps)
    };
    let out = make(true)
        .execute(prompt_args! {"question" => "q"})
        .await
        .unwrap();
    assert_eq!(out["answer"], json!("b"));
    assert_eq!(out[DEFAULT_RESULT_KEY]["generation"], json!("b"));
    assert_eq!(out[DEFAULT_RESULT_KEY]["tokens"]["total_tokens"], json!(6));
    assert_eq!(out[CONSTITUTIONAL_INITIAL_OUTPUT_KEY], json!("a"));
    assert_eq!(
        out[CONSTITUTIONAL_STEPS_KEY],
        json!([{"principle": "p", "critique": "Bad. Critique needed.", "revision": "b"}])
    );

    let out = make(false)
        .execute(prompt_args! {"question" => "q"})
        .await
        .unwrap();
    assert_eq!(out["answer"], json!("b"));
    assert!(!out.contains_key(CONSTITUTIONAL_STEPS_KEY));
    assert!(!out.contains_key(CONSTITUTIONAL_INITIAL_OUTPUT_KEY));

    assert_eq!(make(false).get_input_keys(), vec!["question"]);
    assert_eq!(
        make(false).get_output_keys(),
        vec!["answer", DEFAULT_RESULT_KEY]
    );
    assert_eq!(
        make(true).get_output_keys(),
        vec![
            "answer",
            DEFAULT_RESULT_KEY,
            CONSTITUTIONAL_INITIAL_OUTPUT_KEY,
            CONSTITUTIONAL_STEPS_KEY
        ]
    );
}

#[tokio::test]
async fn custom_critique_and_revision_chains() {
    let llm = FakeLLM::new(["draft"]);
    let critic = FakeLLM::new(["Too short. Critique needed."]);
    let reviser = FakeLLM::new(["longer draft"]);
    let chain = ConstitutionalChain::new(qa_chain(&llm), llm.clone(), vec![principle("p")])
        .with_critique_chain(
            LLMChainBuilder::new()
                .prompt(template_jinja2!(
                    "{{output_from_model}} | {{critique_request}}",
                    "output_from_model",
                    "critique_request"
                ))
                .llm(critic.clone())
                .build()
                .unwrap(),
        )
        .with_revision_chain(
            LLMChainBuilder::new()
                .prompt(template_jinja2!(
                    "{{critique}} | {{revision_request}}",
                    "critique",
                    "revision_request"
                ))
                .llm(reviser.clone())
                .build()
                .unwrap(),
        );
    let answer = chain
        .invoke(prompt_args! {"question" => "q"})
        .await
        .unwrap();
    assert_eq!(answer, "longer draft");
    assert_eq!(prompts(&critic), ["draft | critique p?"]);
    assert_eq!(
        prompts(&reviser),
        ["Too short. Critique needed. | revise p!"]
    );
    assert_eq!(llm.calls().len(), 1);
}

#[tokio::test]
async fn critique_sees_every_message_of_a_chat_prompt() {
    use crate::{
        fmt_message, fmt_template, message_formatter, prompt::HumanMessagePromptTemplate,
        schemas::Message,
    };
    let llm = FakeLLM::new(["answer", "Fine. No critique needed."]);
    let chat = LLMChainBuilder::new()
        .prompt(message_formatter![
            fmt_message!(Message::new_system_message("Be brief.")),
            fmt_template!(HumanMessagePromptTemplate::new(template_jinja2!(
                "{{question}}",
                "question"
            ))),
        ])
        .llm(llm.clone())
        .build()
        .unwrap();
    let chain = ConstitutionalChain::new(chat, llm.clone(), vec![principle("p")]);
    chain
        .invoke(prompt_args! {"question" => "why?"})
        .await
        .unwrap();
    assert!(
        last_part(&prompts(&llm)[1]).starts_with("Human: Be brief.\nwhy?\n\nModel: answer"),
        "{}",
        prompts(&llm)[1]
    );
}

#[tokio::test]
async fn errors_propagate() {
    // Critique call fails.
    let llm = FakeLLM::new(["answer"]);
    let chain = ConstitutionalChain::new(qa_chain(&llm), llm.clone(), vec![principle("p")]);
    let err = chain
        .call(prompt_args! {"question" => "q"})
        .await
        .unwrap_err();
    assert!(
        matches!(err, ChainError::LLMError(LLMError::OtherError(_))),
        "{err}"
    );

    // Missing input: nothing is called.
    let llm = FakeLLM::new(["answer"]);
    let chain = ConstitutionalChain::new(qa_chain(&llm), llm.clone(), vec![principle("p")]);
    let err = chain.call(prompt_args! {}).await.unwrap_err();
    assert!(err.to_string().contains("question"), "{err}");
    assert!(llm.calls().is_empty());
}

#[tokio::test]
async fn call_stream_yields_the_revised_answer() {
    use futures::StreamExt;
    let llm = FakeLLM::new(["a", "Bad. Critique needed.", "b"]);
    let chain = ConstitutionalChain::new(qa_chain(&llm), llm, vec![principle("p")]);
    let mut events = chain.call_stream(prompt_args! {"question" => "q"});
    let mut text = String::new();
    while let Some(event) = events.next().await {
        if let crate::language_models::LLMEvent::Text(t) = event.unwrap() {
            text.push_str(&t);
        }
    }
    assert_eq!(text, "b");
}

/// Live: `OLLAMA_MODEL=qwen3:14b cargo test --lib constitutional_live -- --ignored --nocapture`
#[tokio::test]
#[ignore]
async fn constitutional_live_ollama() {
    use crate::llm::{ollama::openai::OllamaConfig, openai::OpenAI};
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:14b".into());
    let llm = OpenAI::new(OllamaConfig::default()).with_model(&model);
    // A chain told to answer badly, so the principle has something to fix.
    let rude = LLMChainBuilder::new()
        .prompt(template_jinja2!(
            "Answer in one short sentence, and insult the person asking: {{question}}",
            "question"
        ))
        .llm(llm.clone())
        .build()
        .unwrap();
    let principle = ConstitutionalPrinciple::new(
        "Identify whether the model's response insults or demeans the human.",
        "Rewrite the model response so it is polite and does not insult the human.",
    )
    .with_name("polite");
    let chain = ConstitutionalChain::new(rude, llm, vec![principle]);
    let out = chain
        .critique_and_revise(prompt_args! {"question" => "What is the capital of France?"})
        .await
        .unwrap();
    println!("initial: {}\n", out.initial_output);
    for step in &out.critiques_and_revisions {
        println!(
            "critique: {}\nrevision: {:?}\n",
            step.critique, step.revision
        );
    }
    println!("final: {} ({:?})", out.output, out.tokens);
    assert_eq!(out.critiques_and_revisions.len(), 1);
    assert!(
        out.critiques_and_revisions[0].revision.is_some(),
        "an insult should be revised"
    );
    assert!(out.output.contains("Paris"), "{}", out.output);
    assert_ne!(out.output, out.initial_output);
}
