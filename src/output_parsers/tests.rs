use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::{
    typed::{extract_json, strip_reasoning},
    BooleanParser, CommaSeparatedListParser, JsonParser, OutputParser, OutputParserError,
    RegexParser, ResponseSchema, StructuredOutputParser, TypedOutputParser,
};
use crate::{
    chain::{Chain, LLMChainBuilder},
    prompt::HumanMessagePromptTemplate,
    prompt_args, template_fstring,
    test_utils::FakeLLM,
};

fn parsing_error(result: Result<impl std::fmt::Debug, OutputParserError>) -> String {
    match result {
        Err(OutputParserError::ParsingError(msg)) => msg,
        other => panic!("expected a parsing error, got {other:?}"),
    }
}

#[test]
fn strip_reasoning_drops_leading_think_blocks_only() {
    assert_eq!(strip_reasoning("<think>hmm</think>\n\nYes"), "Yes");
    assert_eq!(
        strip_reasoning("  <think>a</think> <think>b</think>ok"),
        "ok"
    );
    // Unclosed or not leading: left alone.
    assert_eq!(strip_reasoning("<think>never ends"), "<think>never ends");
    assert_eq!(
        strip_reasoning("ok <think>x</think>"),
        "ok <think>x</think>"
    );
}

#[test]
fn extract_json_finds_fenced_bare_and_embedded_json() {
    assert_eq!(
        extract_json("```json\n{\"a\": 1}\n```").unwrap(),
        json!({"a": 1})
    );
    assert_eq!(extract_json("```\n[1, 2]\n```").unwrap(), json!([1, 2]));
    assert_eq!(
        extract_json("```json {\"a\":1}```").unwrap(),
        json!({"a": 1})
    );
    assert_eq!(extract_json("  {\"a\": 1}  ").unwrap(), json!({"a": 1}));
    assert_eq!(
        extract_json("\"just a string\"").unwrap(),
        json!("just a string")
    );
    assert_eq!(
        extract_json("Sure! Here it is: {\"a\": {\"b\": 2}} Hope that helps.").unwrap(),
        json!({"a": {"b": 2}})
    );
    assert_eq!(
        extract_json("The list: [\"x\", \"y\"].").unwrap(),
        json!(["x", "y"])
    );
    // A fenced block that isn't JSON is skipped for one that is.
    assert_eq!(
        extract_json("```rust\nfn main() {}\n```\n```json\n{\"ok\": true}\n```").unwrap(),
        json!({"ok": true})
    );
    // Reasoning blocks with braces in them don't confuse the search.
    assert_eq!(
        extract_json("<think>maybe {\"a\": 0}?</think>{\"a\": 1}").unwrap(),
        json!({"a": 1})
    );
    let err = parsing_error(extract_json("no json here"));
    assert!(err.contains("no JSON found"), "{err}");
    assert!(err.contains("\"no json here\""), "{err}");
}

#[test]
fn error_previews_are_cut_on_char_boundaries() {
    let long = "é".repeat(500);
    let err = parsing_error(BooleanParser::new().parse_output(&long));
    assert!(err.ends_with("..."), "{err}");
    assert!(err.chars().filter(|c| *c == 'é').count() == 200, "{err}");
}

#[test]
fn boolean_parser_normalizes_answers() {
    let p = BooleanParser::new();
    for yes in [
        "YES",
        "yes",
        " Yes. ",
        "`true`",
        "\"TRUE\"",
        "'yes'",
        "<think>hmm</think> yes",
    ] {
        assert!(p.parse_output(yes).unwrap(), "{yes:?}");
    }
    for no in ["no", "False", " NO.\n", "`false`"] {
        assert!(!p.parse_output(no).unwrap(), "{no:?}");
    }
    let err = parsing_error(p.parse_output("maybe"));
    assert!(err.contains("\"maybe\"") && err.contains("YES"), "{err}");
    assert!(p.format_instructions().contains("`YES` or `NO`"));
}

#[test]
fn boolean_parser_custom_values() {
    let p = BooleanParser::new().with_values(["si", "Correct"], ["no", "wrong"]);
    assert!(p.parse_output("Sí".replace('í', "i").as_str()).unwrap());
    assert!(p.parse_output("correct").unwrap());
    assert!(!p.parse_output("WRONG").unwrap());
    assert!(p.parse_output("yes").is_err(), "defaults are replaced");
    assert_eq!(p.format_instructions(), "Answer with only `SI` or `NO`.");
}

#[test]
fn comma_list_parser_trims_and_drops_empty_items() {
    let p = CommaSeparatedListParser::new();
    assert_eq!(
        p.parse_output(" red, green ,blue,, \n").unwrap(),
        ["red", "green", "blue"]
    );
    assert_eq!(p.parse_output("<think>x, y</think>one").unwrap(), ["one"]);
    assert!(p.parse_output("  ").unwrap().is_empty());
    assert!(p.format_instructions().contains("comma separated"));
}

#[test]
fn regex_parser_returns_named_groups() {
    let p = RegexParser::new(r"Score: (?P<score>\d+)(?:\s+Reason: (?P<reason>.+))?").unwrap();
    let fields = p
        .parse_output("blah\nScore: 8\nReason: short and clear")
        .unwrap();
    assert_eq!(
        fields,
        HashMap::from([
            ("score".to_string(), "8".to_string()),
            ("reason".to_string(), "short and clear".to_string()),
        ])
    );
    // A group that didn't take part is left out.
    let fields = p.parse_output("Score: 3").unwrap();
    assert_eq!(fields.len(), 1);
    assert_eq!(fields["score"], "3");

    let err = parsing_error(p.parse_output("no score"));
    assert!(err.contains("doesn't match"), "{err}");

    // A score inside the reasoning block is not the answer.
    let fields = p
        .parse_output("<think>Score: 1 seems low</think>Score: 9")
        .unwrap();
    assert_eq!(fields["score"], "9");
    assert!(!fields.contains_key("reason"), "{fields:?}");

    let instructions = p.format_instructions();
    assert!(
        instructions.contains(r"Score: (?P<score>\d+)"),
        "{instructions}"
    );
    assert!(instructions.contains("score, reason"), "{instructions}");
    let p = p.with_format_instructions("Answer like `Score: 5`");
    assert_eq!(p.format_instructions(), "Answer like `Score: 5`");
}

#[test]
fn regex_parser_rejects_bad_patterns() {
    assert!(matches!(
        RegexParser::new("("),
        Err(OutputParserError::RegexError(_))
    ));
    let err = parsing_error(RegexParser::new(r"(\d+)"));
    assert!(err.contains("no named capture groups"), "{err}");
}

fn qa_schema() -> StructuredOutputParser {
    StructuredOutputParser::new([
        ResponseSchema::new("answer", "answer to the user's question"),
        ResponseSchema::new("source", "website used to answer"),
    ])
}

#[test]
fn structured_parser_instructions_list_every_field() {
    let instructions = qa_schema().format_instructions();
    assert!(
        instructions.contains(
            "```json\n{\n\t\"answer\": string // answer to the user's question\n\t\"source\": string // website used to answer\n}\n```"
        ),
        "{instructions}"
    );
}

#[test]
fn structured_parser_checks_required_fields() {
    let p = qa_schema();
    let out = p
        .parse_output("```json\n{\"answer\": \"Paris\", \"source\": \"wiki\", \"n\": 3}\n```")
        .unwrap();
    let mut expected = Map::new();
    expected.insert("answer".into(), json!("Paris"));
    expected.insert("source".into(), json!("wiki"));
    expected.insert("n".into(), json!(3));
    assert_eq!(out, expected);

    // Bare JSON is fine too.
    assert!(p.parse_output("{\"answer\": 1, \"source\": null}").is_ok());

    let err = parsing_error(p.parse_output("{\"answer\": \"Paris\"}"));
    assert!(err.contains("missing the fields [\"source\"]"), "{err}");
    let err = parsing_error(p.parse_output("[1, 2]"));
    assert!(err.contains("expected a JSON object"), "{err}");
    let err = parsing_error(p.parse_output("Paris"));
    assert!(err.contains("no JSON found"), "{err}");
}

#[derive(Debug, Deserialize, PartialEq)]
struct Movie {
    title: String,
    year: u16,
    genres: Vec<String>,
}

#[test]
fn json_parser_deserializes_into_the_target_type() {
    let p = JsonParser::<Movie>::new();
    let movie = p
        .parse_output(
            "<think>let me recall</think>Here you go:\n```json\n{\"title\": \"Alien\", \"year\": 1979, \"genres\": [\"horror\", \"sci-fi\"]}\n```",
        )
        .unwrap();
    assert_eq!(
        movie,
        Movie {
            title: "Alien".into(),
            year: 1979,
            genres: vec!["horror".into(), "sci-fi".into()],
        }
    );
    let err = parsing_error(p.parse_output("{\"title\": \"Alien\"}"));
    assert!(err.contains("missing field `year`"), "{err}");

    // Any DeserializeOwned type works, e.g. a plain list.
    let numbers: Vec<i32> = JsonParser::new().parse_output("[1, 2, 3]").unwrap();
    assert_eq!(numbers, [1, 2, 3]);
}

#[test]
fn json_parser_instructions_include_the_schema() {
    let schema = json!({"type": "object", "properties": {"title": {"type": "string"}}});
    let p = JsonParser::<Movie>::new().with_schema(schema.clone());
    let instructions = p.format_instructions();
    assert!(
        instructions.contains(&serde_json::to_string_pretty(&schema).unwrap()),
        "{instructions}"
    );
    assert!(instructions.contains("```json"), "{instructions}");
    let plain = JsonParser::<Value>::new().format_instructions();
    assert!(plain.contains("JSON value"), "{plain}");
    assert!(!plain.contains("schema"), "{plain}");
}

#[tokio::test]
async fn typed_parsers_work_as_chain_output_parsers() {
    async fn run(parser: Box<dyn OutputParser>, generation: &str) -> String {
        let chain = LLMChainBuilder::new()
            .prompt(HumanMessagePromptTemplate::new(template_fstring!(
                "{q}", "q"
            )))
            .llm(FakeLLM::new([generation.to_string()]))
            .output_parser(parser)
            .build()
            .unwrap();
        chain
            .call(prompt_args! {"q" => "?"})
            .await
            .unwrap()
            .generation
    }

    assert_eq!(run(BooleanParser::new().into(), " Yes.").await, "true");
    assert_eq!(
        run(CommaSeparatedListParser::new().into(), "a, b").await,
        r#"["a","b"]"#
    );
    assert_eq!(
        run(
            RegexParser::new(r"(?P<b>\d) (?P<a>\d)").unwrap().into(),
            "1 2"
        )
        .await,
        r#"{"a":"2","b":"1"}"#
    );
    let structured = run(
        qa_schema().into(),
        "```json\n{\"source\": \"y\", \"answer\": \"x\"}\n```",
    )
    .await;
    assert_eq!(
        serde_json::from_str::<Value>(&structured).unwrap(),
        json!({"answer": "x", "source": "y"})
    );
    let movie = run(
        JsonParser::<Movie>::new().into(),
        "ok: {\"title\": \"Up\", \"year\": 2009, \"genres\": []}",
    )
    .await;
    // Compact JSON (key order depends on serde_json's `preserve_order` feature).
    assert!(!movie.contains(' '), "{movie}");
    assert_eq!(
        serde_json::from_str::<Value>(&movie).unwrap(),
        json!({"title": "Up", "year": 2009, "genres": []})
    );

    // A failed parse is a chain error.
    let chain = LLMChainBuilder::new()
        .prompt(HumanMessagePromptTemplate::new(template_fstring!(
            "{q}", "q"
        )))
        .llm(FakeLLM::new(["{\"title\": \"Up\"}"]))
        .output_parser(JsonParser::<Movie>::new())
        .build()
        .unwrap();
    let err = chain.call(prompt_args! {"q" => "?"}).await.unwrap_err();
    assert!(err.to_string().contains("missing field `year`"), "{err}");
    let chain = LLMChainBuilder::new()
        .prompt(HumanMessagePromptTemplate::new(template_fstring!(
            "{q}", "q"
        )))
        .llm(FakeLLM::new(["perhaps"]))
        .output_parser(BooleanParser::new())
        .build()
        .unwrap();
    let err = chain.call(prompt_args! {"q" => "?"}).await.unwrap_err();
    assert!(err.to_string().contains("perhaps"), "{err}");
}

#[test]
fn parsers_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BooleanParser>();
    assert_send_sync::<CommaSeparatedListParser>();
    assert_send_sync::<RegexParser>();
    assert_send_sync::<StructuredOutputParser>();
    assert_send_sync::<JsonParser<std::rc::Rc<u8>>>();
}
