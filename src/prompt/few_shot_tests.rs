//! Offline tests for few-shot prompts and example selectors.
use std::sync::{Arc, Mutex as StdMutex};

use async_trait::async_trait;
use serde_json::json;

use crate::{
    chain::{Chain, LLMChainBuilder},
    embedding::{Embedder, EmbedderError},
    prompt::{
        ExampleSelector, FewShotPromptTemplate, FormatPrompter, LengthBasedExampleSelector,
        PromptArgs, PromptError, SemanticSimilarityExampleSelector,
    },
    prompt_args, template_fstring,
    test_utils::{FakeEmbedder, FakeLLM},
};

fn antonym_prompt() -> crate::prompt::PromptTemplate {
    template_fstring!("Word: {word}\nAntonym: {antonym}", "word", "antonym")
}

fn antonyms() -> Vec<PromptArgs> {
    vec![
        prompt_args! {"word" => "happy", "antonym" => "sad"},
        prompt_args! {"word" => "tall", "antonym" => "short"},
    ]
}

fn words(examples: &[PromptArgs]) -> Vec<String> {
    examples
        .iter()
        .map(|e| e["word"].as_str().unwrap().to_string())
        .collect()
}

#[tokio::test]
async fn few_shot_formats_prefix_examples_and_suffix() {
    let prompt = FewShotPromptTemplate::new(antonym_prompt(), antonyms())
        .with_prefix(template_fstring!("Give the antonym, {name}.", "name"))
        .with_suffix(template_fstring!("Word: {input}\nAntonym:", "input"));

    let expected = "Give the antonym, Ana.\n\n\
                    Word: happy\nAntonym: sad\n\n\
                    Word: tall\nAntonym: short\n\n\
                    Word: big\nAntonym:";
    let args = prompt_args! {"name" => "Ana", "input" => "big"};
    assert_eq!(prompt.format(args.clone()).await.unwrap(), expected);

    // Sync and async FormatPrompter agree, as one human message.
    for value in [
        prompt.format_prompt(args.clone()).unwrap(),
        prompt.format_prompt_async(args).await.unwrap(),
    ] {
        let messages = value.to_chat_messages();
        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0].message_type,
            crate::schemas::MessageType::HumanMessage
        );
        assert_eq!(messages[0].content, expected);
    }

    // Input variables are the prefix's and suffix's, deduplicated, in order.
    let prompt = prompt.with_suffix(template_fstring!("{input} {name}", "input", "name"));
    assert_eq!(prompt.get_input_variables(), vec!["name", "input"]);
}

#[tokio::test]
async fn few_shot_edge_cases() {
    // Prefix only / suffix only / custom separator.
    let prefix_only = FewShotPromptTemplate::new(antonym_prompt(), vec![])
        .with_prefix(template_fstring!("This is a {foo} test.", "foo"));
    assert_eq!(
        prefix_only
            .format(prompt_args! {"foo" => "bar"})
            .await
            .unwrap(),
        "This is a bar test."
    );
    let suffix_sep = FewShotPromptTemplate::new(antonym_prompt(), antonyms())
        .with_suffix(template_fstring!("{input}", "input"))
        .with_separator("\n---\n");
    assert_eq!(
        suffix_sep
            .format(prompt_args! {"input" => "x"})
            .await
            .unwrap(),
        "Word: happy\nAntonym: sad\n---\nWord: tall\nAntonym: short\n---\nx"
    );

    // Empty pieces are skipped, so no doubled separators.
    let empty_prefix = FewShotPromptTemplate::new(antonym_prompt(), antonyms()[..1].to_vec())
        .with_prefix(template_fstring!("",))
        .with_suffix(template_fstring!("end",));
    assert_eq!(
        empty_prefix.format(prompt_args! {}).await.unwrap(),
        "Word: happy\nAntonym: sad\n\nend"
    );

    // Braces in examples are not treated as input variables.
    let braces = FewShotPromptTemplate::new(
        template_fstring!("{code}", "code"),
        vec![prompt_args! {"code" => "fn f() {input}"}],
    )
    .with_suffix(template_fstring!("{input}", "input"));
    assert_eq!(
        braces.format(prompt_args! {"input" => "Q"}).await.unwrap(),
        "fn f() {input}\n\nQ"
    );

    // Missing input variable / bad example.
    let err = prefix_only.format(prompt_args! {}).await.unwrap_err();
    assert!(matches!(err, PromptError::MissingVariable(v) if v == "foo"));
    let bad = FewShotPromptTemplate::new(
        antonym_prompt(),
        vec![antonyms()[0].clone(), prompt_args! {"word" => "x"}],
    );
    let err = bad.format(prompt_args! {}).await.unwrap_err();
    assert!(
        err.to_string().contains("example 2") && err.to_string().contains("antonym"),
        "{err}"
    );
}

#[tokio::test]
async fn length_selector_fits_examples_within_max_length() {
    // Each formatted example is 4 words ("Word: happy Antonym: sad").
    let examples = vec![
        prompt_args! {"word" => "happy", "antonym" => "sad"},
        prompt_args! {"word" => "tall", "antonym" => "short"},
        prompt_args! {"word" => "energetic", "antonym" => "lethargic"},
    ];
    let selector = LengthBasedExampleSelector::new(antonym_prompt(), examples)
        .unwrap()
        .with_max_length(10);

    // 1-word input: 9 left, two examples (8) fit, the third doesn't.
    let picked = selector
        .select_examples(&prompt_args! {"input" => "big"})
        .await
        .unwrap();
    assert_eq!(words(&picked), ["happy", "tall"]);

    // Exactly fitting: 2 + 8 = 10.
    let picked = selector
        .select_examples(&prompt_args! {"input" => "very big"})
        .await
        .unwrap();
    assert_eq!(words(&picked), ["happy", "tall"]);

    // Longer input leaves room for fewer examples; non-string inputs count too.
    let picked = selector
        .select_examples(&prompt_args! {"input" => "a b c", "n" => 1})
        .await
        .unwrap();
    assert_eq!(words(&picked), ["happy"]);
    let picked = selector
        .select_examples(&prompt_args! {"input" => "one two three four five six seven"})
        .await
        .unwrap();
    assert!(picked.is_empty());
    let picked = selector
        .select_examples(&prompt_args! {"input" => "w ".repeat(50)})
        .await
        .unwrap();
    assert!(picked.is_empty());

    // Stops at the first example that doesn't fit, keeping order.
    selector
        .add_example(prompt_args! {"word" => "a", "antonym" => "b"})
        .await
        .unwrap();
    let picked = selector.select_examples(&prompt_args! {}).await.unwrap();
    assert_eq!(words(&picked), ["happy", "tall"]);
    let texts = LengthBasedExampleSelector::new(
        template_fstring!("{text}", "text"),
        ["a b c", "d e f g h", "i"]
            .into_iter()
            .map(|t| prompt_args! {"text" => t})
            .collect(),
    )
    .unwrap()
    .with_max_length(5);
    // "i" would fit after skipping the 5-word one, but selection stops there.
    let picked = texts.select_examples(&prompt_args! {}).await.unwrap();
    assert_eq!(picked, vec![prompt_args! {"text" => "a b c"}]);

    // Default max length takes them all.
    let all = LengthBasedExampleSelector::new(antonym_prompt(), antonyms()).unwrap();
    assert_eq!(
        all.select_examples(&prompt_args! {}).await.unwrap().len(),
        2
    );

    // Custom length function (characters), applied to existing examples too.
    let by_chars = LengthBasedExampleSelector::new(antonym_prompt(), antonyms())
        .unwrap()
        .with_max_length(50)
        .with_length_fn(|s: &str| s.chars().count());
    // "Word: happy\nAntonym: sad" is 24 chars, "Word: tall\nAntonym: short" 25.
    let picked = by_chars.select_examples(&prompt_args! {}).await.unwrap();
    assert_eq!(words(&picked), ["happy", "tall"]);
    let picked = by_chars
        .select_examples(&prompt_args! {"input" => "xy"})
        .await
        .unwrap();
    assert_eq!(words(&picked), ["happy"]);
    // Exactly at the limit still fits: 1 + 24 + 25 = 50.
    let picked = by_chars
        .select_examples(&prompt_args! {"input" => "x"})
        .await
        .unwrap();
    assert_eq!(words(&picked), ["happy", "tall"]);

    // Examples must match the example prompt.
    let err = LengthBasedExampleSelector::new(antonym_prompt(), vec![prompt_args! {"word" => "x"}])
        .unwrap_err();
    assert!(matches!(err, PromptError::MissingVariable(v) if v == "antonym"));
    let err = all
        .add_example(prompt_args! {"antonym" => "x"})
        .await
        .unwrap_err();
    assert!(matches!(err, PromptError::MissingVariable(v) if v == "word"));
}

fn weather_embedder() -> FakeEmbedder {
    FakeEmbedder::new(
        3,
        &[
            ("sunny", vec![1.0, 0.0, 0.0]),
            ("rainy", vec![0.9, 0.1, 0.0]),
            ("windy", vec![0.6, 0.4, 0.0]),
            ("pizza", vec![0.0, 1.0, 0.0]),
            ("pasta", vec![0.1, 0.9, 0.0]),
            ("hot", vec![1.0, 0.05, 0.0]),
            ("dinner", vec![0.0, 1.0, 0.1]),
        ],
    )
}

fn weather_examples() -> Vec<PromptArgs> {
    ["sunny", "pizza", "rainy", "pasta", "windy"]
        .into_iter()
        .map(|w| prompt_args! {"word" => w})
        .collect()
}

#[tokio::test]
async fn similarity_selector_picks_the_k_closest() {
    let selector =
        SemanticSimilarityExampleSelector::from_examples(weather_examples(), weather_embedder(), 2)
            .await
            .unwrap();
    let picked = selector
        .select_examples(&prompt_args! {"word" => "hot"})
        .await
        .unwrap();
    assert_eq!(words(&picked), ["sunny", "rainy"]);
    let picked = selector
        .select_examples(&prompt_args! {"word" => "dinner"})
        .await
        .unwrap();
    assert_eq!(words(&picked), ["pizza", "pasta"]);

    // k larger than the pool returns all, most similar first.
    let all = selector.with_k(10);
    let picked = all
        .select_examples(&prompt_args! {"word" => "hot"})
        .await
        .unwrap();
    assert_eq!(
        words(&picked),
        ["sunny", "rainy", "windy", "pasta", "pizza"]
    );

    // k = 0 and an empty pool select nothing.
    let none = all.with_k(0);
    assert!(none
        .select_examples(&prompt_args! {"word" => "hot"})
        .await
        .unwrap()
        .is_empty());
    let empty = SemanticSimilarityExampleSelector::new(weather_embedder());
    assert!(empty
        .select_examples(&prompt_args! {"word" => "hot"})
        .await
        .unwrap()
        .is_empty());

    // add_example joins the pool.
    empty
        .add_example(prompt_args! {"word" => "pizza"})
        .await
        .unwrap();
    empty
        .add_example(prompt_args! {"word" => "sunny"})
        .await
        .unwrap();
    let picked = empty
        .with_k(1)
        .select_examples(&prompt_args! {"word" => "hot"})
        .await
        .unwrap();
    assert_eq!(words(&picked), ["sunny"]);
}

/// Embedder that records what it was asked to embed.
#[derive(Clone, Default)]
struct RecordingEmbedder {
    texts: Arc<StdMutex<Vec<String>>>,
    calls: Arc<StdMutex<usize>>,
    broken: bool,
}

#[async_trait]
impl Embedder for RecordingEmbedder {
    async fn embed_documents(&self, documents: &[String]) -> Result<Vec<Vec<f64>>, EmbedderError> {
        *self.calls.lock().unwrap() += 1;
        self.texts.lock().unwrap().extend(documents.iter().cloned());
        if self.broken {
            return Ok(vec![]);
        }
        Ok(documents.iter().map(|_| vec![1.0, 0.0]).collect())
    }
    async fn embed_query(&self, text: &str) -> Result<Vec<f64>, EmbedderError> {
        self.texts.lock().unwrap().push(text.to_string());
        if self.broken {
            return Err(EmbedderError::HttpError {
                status_code: reqwest::StatusCode::SERVICE_UNAVAILABLE,
                error_message: "down".into(),
            });
        }
        Ok(vec![0.0, 0.0])
    }
}

#[tokio::test]
async fn similarity_selector_embeds_input_keys_in_one_batch() {
    let embedder = RecordingEmbedder::default();
    let selector = SemanticSimilarityExampleSelector::new(embedder.clone())
        .with_input_keys(["question", "context"]);
    selector
        .add_examples(vec![
            prompt_args! {"question" => "q1", "answer" => "a1", "context" => "c1"},
            prompt_args! {"question" => "q2", "answer" => "a2", "n" => json!(3)},
        ])
        .await
        .unwrap();
    selector.add_examples(vec![]).await.unwrap();
    assert_eq!(*embedder.calls.lock().unwrap(), 1);

    // A zero query vector is similarity 0 everywhere: ties keep insertion order.
    let picked = selector
        .select_examples(&prompt_args! {"question" => "q?", "answer" => "ignored"})
        .await
        .unwrap();
    assert_eq!(picked.len(), 2);
    assert_eq!(picked[0]["question"], "q1");
    // Only the input keys, sorted by key; answers never embedded.
    assert_eq!(*embedder.texts.lock().unwrap(), ["c1 q1", "q2", "q?"]);

    // Without input keys every value counts, sorted by key, strings unquoted.
    let embedder = RecordingEmbedder::default();
    let all_keys = SemanticSimilarityExampleSelector::new(embedder.clone());
    all_keys
        .add_example(prompt_args! {"b" => "two", "a" => "one", "c" => json!(3)})
        .await
        .unwrap();
    assert_eq!(*embedder.texts.lock().unwrap(), ["one two 3"]);

    // Embedder failures are prompt errors.
    let broken = SemanticSimilarityExampleSelector::new(RecordingEmbedder {
        broken: true,
        ..Default::default()
    });
    let err = broken
        .add_example(prompt_args! {"a" => "x"})
        .await
        .unwrap_err();
    assert!(err.to_string().contains("0 vectors for 1"), "{err}");

    // A failing query embedding is a prompt error too (needs a non-empty pool).
    let failing_query = SemanticSimilarityExampleSelector::new(RecordingEmbedder {
        broken: true,
        ..Default::default()
    });
    failing_query.examples_for_test(vec![(prompt_args! {"a" => "x"}, vec![1.0])]);
    let err = failing_query
        .select_examples(&prompt_args! {"a" => "y"})
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("embedding the input") && err.to_string().contains("down"),
        "{err}"
    );
}

#[tokio::test]
async fn few_shot_with_selector_in_a_chain() {
    let selector = Arc::new(
        SemanticSimilarityExampleSelector::from_examples(weather_examples(), weather_embedder(), 1)
            .await
            .unwrap(),
    );
    let prompt = FewShotPromptTemplate::with_shared_selector(
        template_fstring!("Example: {word}", "word"),
        selector.clone(),
    )
    .with_suffix(template_fstring!("Now: {word}", "word"));

    // The sync path can't run a selector.
    let err = prompt
        .format_prompt(prompt_args! {"word" => "hot"})
        .unwrap_err();
    assert!(err.to_string().contains("format_prompt_async"), "{err}");

    let llm = FakeLLM::new(["warm", "Italian", "hot again"]);
    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(llm.clone())
        .build()
        .unwrap();
    assert_eq!(chain.get_input_keys(), vec!["word"]);
    chain.invoke(prompt_args! {"word" => "hot"}).await.unwrap();
    chain
        .invoke(prompt_args! {"word" => "dinner"})
        .await
        .unwrap();

    // Examples added through the shared handle are used right away.
    selector
        .add_example(prompt_args! {"word" => "hot"})
        .await
        .unwrap();
    chain.call(prompt_args! {"word" => "hot"}).await.unwrap();

    let calls = llm.calls();
    assert_eq!(calls[0][0].content, "Example: sunny\n\nNow: hot");
    assert_eq!(calls[1][0].content, "Example: pizza\n\nNow: dinner");
    assert_eq!(calls[2][0].content, "Example: hot\n\nNow: hot");
}

#[tokio::test]
async fn length_selector_in_few_shot_prompt() {
    let selector = LengthBasedExampleSelector::new(antonym_prompt(), antonyms())
        .unwrap()
        .with_max_length(6);
    let prompt = FewShotPromptTemplate::with_selector(antonym_prompt(), selector)
        .with_suffix(template_fstring!("Word: {input}\nAntonym:", "input"));
    // 1-word input leaves 5: only the first example (4 words) fits.
    assert_eq!(
        prompt
            .format(prompt_args! {"input" => "big"})
            .await
            .unwrap(),
        "Word: happy\nAntonym: sad\n\nWord: big\nAntonym:"
    );
    assert!(format!("{prompt:?}").contains("selector"));
}

/// Live: `OLLAMA_MODEL=qwen3:14b cargo test --lib --all-features few_shot_live -- --ignored --nocapture`
///
/// Real embeddings (nomic-embed-text) pick the examples, and the model follows the
/// made-up output format they show.
#[cfg(feature = "ollama")]
#[tokio::test]
#[ignore]
async fn few_shot_live_similarity_examples_steer_the_model() {
    use crate::{
        embedding::OllamaEmbedder,
        llm::{ollama::openai::OllamaConfig, OpenAI},
        output_parsers::strip_reasoning,
    };

    let examples = vec![
        prompt_args! {"input" => "apple", "output" => "FRUIT::apple::red"},
        prompt_args! {"input" => "banana", "output" => "FRUIT::banana::yellow"},
        prompt_args! {"input" => "carrot", "output" => "VEGETABLE::carrot::orange"},
        prompt_args! {"input" => "spinach", "output" => "VEGETABLE::spinach::green"},
        prompt_args! {"input" => "truck", "output" => "VEHICLE::truck::any"},
        prompt_args! {"input" => "bicycle", "output" => "VEHICLE::bicycle::any"},
    ];
    let selector =
        SemanticSimilarityExampleSelector::from_examples(examples, OllamaEmbedder::default(), 2)
            .await
            .unwrap()
            .with_input_keys(["input"]);

    let picked = selector
        .select_examples(&prompt_args! {"input" => "motorcycle"})
        .await
        .unwrap();
    println!("picked for motorcycle: {picked:?}");
    assert!(picked
        .iter()
        .all(|e| e["output"].as_str().unwrap().starts_with("VEHICLE")));

    let prompt = FewShotPromptTemplate::with_selector(
        template_fstring!("Input: {input}\nOutput: {output}", "input", "output"),
        selector,
    )
    .with_prefix(template_fstring!(
        "Classify the input exactly like the examples. Reply with the output line only.",
    ))
    .with_suffix(template_fstring!("Input: {input}\nOutput:", "input"));

    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:14b".into());
    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(OpenAI::new(OllamaConfig::default()).with_model(&model))
        .build()
        .unwrap();
    for (input, expected) in [
        ("motorcycle", "VEHICLE::motorcycle"),
        ("pear", "FRUIT::pear"),
    ] {
        let answer = chain.invoke(prompt_args! {"input" => input}).await.unwrap();
        let answer = strip_reasoning(&answer).trim().to_string();
        println!("{input}: {answer}");
        assert!(answer.starts_with(expected), "{input}: {answer}");
    }
}
