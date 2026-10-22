use serde_json::json;

use super::*;
use crate::{prompt_args, template_jinja2, test_utils::FakeLLM};

fn prompts(llm: &FakeLLM) -> Vec<String> {
    llm.calls().iter().map(|c| c[0].content.clone()).collect()
}

#[test]
fn parse_math_output_formats() {
    let cases: [(&str, &str, Option<&str>); 11] = [
        ("```text\n37593 * 67\n```", "2518731", Some("37593 * 67")),
        // langchaingo's tag, Python, and no tag at all.
        ("```starlark\n2 ** 10\n```", "1024", Some("2 ** 10")),
        ("```python\nmath.sqrt(16)\n```", "4", Some("math.sqrt(16)")),
        ("```\n1 / 3\n```", "0.333333333333", Some("1 / 3")),
        // Text around the block, inline block, missing closing fence.
        (
            "Sure! Here it is:\n```text\n(40 + 3) * 10000 / 7324.3\n```\nDone.",
            "58.7086820584",
            Some("(40 + 3) * 10000 / 7324.3"),
        ),
        ("```text 7 % 4```", "3", Some("7 % 4")),
        ("```text\n6 * 7", "42", Some("6 * 7")),
        (
            "<think>```text\n1 + 1\n``` no wait</think>\n```text\n2 + 2\n```",
            "4",
            Some("2 + 2"),
        ),
        // The model answered directly.
        ("Answer: 42", "42", None),
        ("Question: x\n```text\n```\nAnswer:  seven ", "seven", None),
        // The last `Answer:` is the final one.
        ("Answer: maybe 6?\nWait.\nAnswer: 7", "7", None),
    ];
    for (output, answer, expression) in cases {
        assert_eq!(
            parse_math_output(output).unwrap(),
            (answer.to_string(), expression.map(String::from)),
            "{output:?}"
        );
    }
}

#[test]
fn parse_math_output_errors() {
    let err = parse_math_output("```text\n2 +* 3\n```")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("evaluating `2 +* 3`") && err.contains("unexpected"),
        "{err}"
    );

    let err = parse_math_output("```text\nimport os\n```")
        .unwrap_err()
        .to_string();
    assert!(err.contains("unknown name"), "{err}");

    let err = parse_math_output("```text\n1 / 0\n```")
        .unwrap_err()
        .to_string();
    assert!(err.contains("division by zero"), "{err}");

    for output in ["I think it's about 12.", "Answer:", "```text\n```"] {
        let err = parse_math_output(output).unwrap_err().to_string();
        assert!(err.contains("expected an expression"), "{output:?}: {err}");
    }
}

#[tokio::test]
async fn solve_evaluates_the_models_expression() {
    let llm = FakeLLM::new(["```text\n(40 + 3) * 10000 / 7324.3\n```"]);
    let chain = LLMMathChain::new(llm.clone());
    let out = chain
        .solve(prompt_args! {"question" => "forty plus three, times ten thousand over 7324.3"})
        .await
        .unwrap();
    assert_eq!(out.answer, "58.7086820584");
    assert_eq!(out.expression.as_deref(), Some("(40 + 3) * 10000 / 7324.3"));
    assert_eq!(out.tokens.unwrap().total_tokens, 2);

    let prompt = &prompts(&llm)[0];
    assert!(prompt.starts_with("Translate a math problem into a single-line arithmetic expression"));
    assert!(prompt.contains("Question: What is 37593 * 67?\n```text\n37593 * 67\n```"));
    assert!(prompt
        .contains("Question: What is the fifth root of 37593?\n```text\n37593 ^ (1 / 5)\n```"));
    // The few-shot expressions are ones the calculator accepts.
    assert_eq!(evaluate_expression("37593 * 67").unwrap(), 2518731.0);
    assert!((evaluate_expression("37593 ^ (1 / 5)").unwrap() - 8.22283).abs() < 1e-4);
    assert!(prompt.ends_with("Question: forty plus three, times ten thousand over 7324.3\n"));
    assert!(!prompt.contains("{{"), "every variable is filled");
}

#[tokio::test]
async fn invoke_and_execute() {
    let make = || LLMMathChain::new(FakeLLM::new(["```text\n2 ^ 10\n```"]));
    assert_eq!(
        make()
            .invoke(prompt_args! {"question" => "q"})
            .await
            .unwrap(),
        "1024"
    );

    let out = make()
        .with_output_key("result")
        .execute(prompt_args! {"question" => "q"})
        .await
        .unwrap();
    assert_eq!(out["result"], json!("1024"));
    assert_eq!(out[LLM_MATH_EXPRESSION_KEY], json!("2 ^ 10"));
    assert_eq!(out[DEFAULT_RESULT_KEY]["generation"], json!("1024"));
    assert!(!out.contains_key(LLM_MATH_DEFAULT_OUTPUT_KEY));

    // A direct answer has no expression.
    let out = LLMMathChain::new(FakeLLM::new(["Answer: 3"]))
        .execute(prompt_args! {"question" => "q"})
        .await
        .unwrap();
    assert_eq!(out[LLM_MATH_DEFAULT_OUTPUT_KEY], json!("3"));
    assert!(!out.contains_key(LLM_MATH_EXPRESSION_KEY));

    let chain = make().with_output_key("result");
    assert_eq!(chain.get_input_keys(), vec!["question"]);
    assert_eq!(
        chain.get_output_keys(),
        vec!["result", DEFAULT_RESULT_KEY, LLM_MATH_EXPRESSION_KEY]
    );
}

#[tokio::test]
async fn custom_input_key() {
    let llm = FakeLLM::new(["```text\n1 + 2\n```"]);
    let chain = LLMMathChain::new(llm.clone()).with_input_key("problem");
    assert_eq!(chain.get_input_keys(), vec!["problem"]);
    let answer = chain
        .invoke(prompt_args! {"problem" => "one plus two"})
        .await
        .unwrap();
    assert_eq!(answer, "3");
    assert!(prompts(&llm)[0].ends_with("Question: one plus two\n"));

    let llm = FakeLLM::new(["```text\n1\n```"]);
    let err = LLMMathChain::new(llm.clone())
        .with_input_key("problem")
        .invoke(prompt_args! {"question" => "q"})
        .await
        .unwrap_err();
    assert!(
        matches!(err, ChainError::MissingInputVariable(ref k) if k == "problem"),
        "{err}"
    );
    assert!(llm.calls().is_empty());
}

#[tokio::test]
async fn from_chain_and_errors() {
    let llm = FakeLLM::new(["```\n3 * 3\n```"]);
    let custom = LLMChainBuilder::new()
        .prompt(template_jinja2!("Expression for: {{question}}", "question"))
        .llm(llm.clone())
        .build()
        .unwrap();
    let chain = LLMMathChain::from_chain(custom);
    assert_eq!(
        chain
            .invoke(prompt_args! {"question" => "nine"})
            .await
            .unwrap(),
        "9"
    );
    assert_eq!(prompts(&llm), ["Expression for: nine"]);

    // Missing question: the prompt fails before any call.
    let llm = FakeLLM::new(["x"]);
    let err = LLMMathChain::new(llm.clone())
        .invoke(prompt_args! {})
        .await
        .unwrap_err();
    assert!(err.to_string().contains("question"), "{err}");
    assert!(llm.calls().is_empty());

    // LLM failure and unparseable output.
    let err = LLMMathChain::new(FakeLLM::new(Vec::<String>::new()))
        .invoke(prompt_args! {"question" => "q"})
        .await
        .unwrap_err();
    assert!(matches!(err, ChainError::LLMError(_)), "{err}");
    let err = LLMMathChain::new(FakeLLM::new(["no idea"]))
        .invoke(prompt_args! {"question" => "q"})
        .await
        .unwrap_err();
    assert!(err.to_string().contains("\"no idea\""), "{err}");
}

/// Live: `OLLAMA_MODEL=qwen3:14b cargo test --lib llm_math_live -- --ignored --nocapture`
#[tokio::test]
#[ignore]
async fn llm_math_live_ollama() {
    use crate::llm::{ollama::openai::OllamaConfig, openai::OpenAI};
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:14b".into());
    let chain = LLMMathChain::new(OpenAI::new(OllamaConfig::default()).with_model(&model));
    let cases = [
        // langchaingo's test question.
        (
            "what is forty plus three? take that then multiply it by ten thousand divided by 7324.3",
            "58.708",
        ),
        ("What is 13 raised to the 0.3432 power?", "2.4116"),
        ("A pizza has 8 slices. 3 friends eat 2 slices each. How many are left?", "2"),
    ];
    for (question, expected) in cases {
        let out = chain
            .solve(prompt_args! {"question" => question})
            .await
            .unwrap();
        println!("{question}\n  {:?} = {}\n", out.expression, out.answer);
        assert!(
            out.answer.starts_with(expected),
            "{question}: {}",
            out.answer
        );
    }
}
