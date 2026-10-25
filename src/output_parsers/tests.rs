use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::{
    typed::{extract_json, strip_reasoning},
    BooleanParser, CombiningParser, CommaSeparatedListParser, JsonParser, OutputParser,
    OutputParserError, RegexDictParser, RegexParser, ResponseSchema, StructuredOutputParser,
    TypedOutputParser,
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
    assert_eq!(
        run(
            RegexDictParser::new([("b", "B"), ("a", "A")]).into(),
            "B: 1\nA: 2"
        )
        .await,
        r#"{"a":"2","b":"1"}"#
    );
    assert_eq!(
        run(
            CombiningParser::new(
                RegexDictParser::new([("z", "Z")]),
                JsonParser::<Value>::new()
            )
            .into(),
            "Z: 1\n\n{\"a\": [true]}"
        )
        .await,
        r#"{"a":[true],"z":"1"}"#
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
    assert_send_sync::<RegexDictParser>();
    assert_send_sync::<CombiningParser>();
}

// The text from langchaingo's `TestRegexDict`.
const REGEX_DICT_TEXT: &str = r#"We have just received a new result from the LLM, and our next step is
to filter and read its format using regular expressions to identify specific fields,
such as:

- Action: Search
- Action Input: How to use this class?
- Additional Fields: "N/A"

To exclude irrelevant information from our return dictionary, we can instruct the LLM to
use a specific command that notifies us when it doesn't know the answer."#;

fn action_dict() -> RegexDictParser {
    RegexDictParser::new([("action", "Action"), ("action_input", "Action Input")])
}

fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn regex_dict_parser_reads_each_label() {
    assert_eq!(
        action_dict().parse_output(REGEX_DICT_TEXT).unwrap(),
        map(&[
            ("action", "Search"),
            ("action_input", "How to use this class?")
        ])
    );
    // "Action:" is not found inside "Action Input:", and values are trimmed.
    assert_eq!(
        action_dict()
            .parse_output("Action Input:  rust 2024  \nAction: search")
            .unwrap(),
        map(&[("action", "search"), ("action_input", "rust 2024")])
    );
    // A value ends at the first sentence-ending period (kept) or the end of the line;
    // periods inside numbers and URLs don't end it.
    assert_eq!(
        RegexDictParser::new([("a", "A")])
            .parse_output("A: one. two\nB: x")
            .unwrap(),
        map(&[("a", "one.")])
    );
    assert_eq!(
        RegexDictParser::new([("a", "A"), ("b", "B")])
            .parse_output("A: 37.4 million (2023).\tMore text.\nB: see https://x.org/a.html")
            .unwrap(),
        map(&[
            ("a", "37.4 million (2023)."),
            ("b", "see https://x.org/a.html")
        ])
    );
    assert_eq!(
        RegexDictParser::new([("a", "A")])
            .parse_output("A: done.\n")
            .unwrap(),
        map(&[("a", "done.")])
    );
    assert_eq!(
        RegexDictParser::new([("a", "A")])
            .parse_output("A:\nB: x")
            .unwrap(),
        map(&[("a", "")])
    );
    // Reasoning is skipped.
    assert_eq!(
        RegexDictParser::new([("name", "Name"), ("age", "Age")])
            .parse_output("<think>Name: no</think>Name: John\nAge: 30")
            .unwrap(),
        map(&[("name", "John"), ("age", "30")])
    );
}

#[test]
fn regex_dict_parser_skips_the_no_update_value() {
    assert_eq!(
        action_dict()
            .with_no_update_value("Search")
            .parse_output(REGEX_DICT_TEXT)
            .unwrap(),
        map(&[("action_input", "How to use this class?")])
    );
    assert_eq!(
        RegexDictParser::new([("a", "A"), ("b", "B")])
            .with_no_update_value("N/A")
            .parse_output("A: N/A\nB: 2")
            .unwrap(),
        map(&[("b", "2")])
    );
}

#[test]
fn regex_dict_parser_labels_are_literal() {
    // langchaingo puts the label into the regex as is; here `.`, `(` or `?` are literal.
    let parser = RegexDictParser::new([("cost", "Cost (USD)?")]);
    assert_eq!(
        parser.parse_output("Cost (USD)?: 12\n").unwrap(),
        map(&[("cost", "12")])
    );
    let err = parsing_error(parser.parse_output("Cost USD: 12"));
    assert!(err.contains("no `Cost (USD)?:`"), "{err}");
    let dotted = RegexDictParser::new([("v", "a.b")]);
    assert!(dotted.parse_output("axb: 1").is_err());
}

#[test]
fn regex_dict_parser_needs_each_label_exactly_once() {
    let err = parsing_error(action_dict().parse_output("Action: search"));
    assert!(err.contains("no `Action Input:`"), "{err}");
    assert!(err.contains("Action: search"), "{err}");
    let err = parsing_error(action_dict().parse_output("Action: a\nAction Input: x\nAction: b"));
    assert!(err.contains("`Action:` appears more than once"), "{err}");
}

#[test]
fn regex_dict_parser_instructions_list_the_labels() {
    assert_eq!(
        action_dict().format_instructions(),
        "Your answer must contain these lines, each once:\nAction: <value>\nAction Input: <value>"
    );
    assert!(action_dict()
        .with_no_update_value("N/A")
        .format_instructions()
        .ends_with("\nWrite N/A as the value you don't know."));
}

// The text from langchaingo's `TestCombine`.
const COMBINE_TEXT: &str = "```json\n{\n    \"answer\": \"Paris\",\n    \"source\": \"https://en.wikipedia.org/wiki/France\"\n}\n```\n\n//Confidence: A, Explanation: Paris is the capital of France according to Wikipedia.";

fn confidence() -> RegexParser {
    RegexParser::new(r"Confidence: (?P<confidence>A|B|C), Explanation: (?P<explanation>.*)")
        .unwrap()
}

#[test]
fn combining_parser_merges_each_part() {
    let parser = CombiningParser::new(
        StructuredOutputParser::new([
            ResponseSchema::new("answer", "The answer to the question"),
            ResponseSchema::new("source", "A link to the source"),
        ]),
        confidence(),
    );
    assert_eq!(
        Value::Object(parser.parse_output(COMBINE_TEXT).unwrap()),
        json!({
            "answer": "Paris",
            "source": "https://en.wikipedia.org/wiki/France",
            "confidence": "A",
            "explanation": "Paris is the capital of France according to Wikipedia.",
        })
    );

    let three = CombiningParser::new(
        RegexDictParser::new([("key1", "Key1")]),
        RegexDictParser::new([("key2", "Key2")]),
    )
    .with(JsonParser::<Value>::new());
    assert_eq!(
        Value::Object(
            three
                .parse_output(
                    "<think>a\n\nb</think>\n\nKey1: value1\r\n  \r\nKey2: value2\n\n{\"n\": 3}\n\n\n"
                )
                .unwrap()
        ),
        json!({"key1": "value1", "key2": "value2", "n": 3})
    );
}

#[test]
fn combining_parser_later_parts_win_on_shared_keys() {
    let parser = CombiningParser::new(
        RegexDictParser::new([("k", "K")]),
        JsonParser::<Value>::new(),
    );
    assert_eq!(
        Value::Object(parser.parse_output("K: first\n\n{\"k\": 2}").unwrap()),
        json!({"k": 2})
    );
}

#[test]
fn combining_parser_errors() {
    let parser = CombiningParser::new(confidence(), RegexDictParser::new([("k", "K")]));
    let err = parsing_error(parser.parse_output("\n\n\n\n"));
    assert!(
        err.contains("expected 2 parts separated by blank lines, got 0"),
        "{err}"
    );
    let err = parsing_error(parser.parse_output("K: 1"));
    assert!(err.contains("got 1"), "{err}");
    let err = parsing_error(parser.parse_output("a\n\nb\n\nc"));
    assert!(err.contains("got 3"), "{err}");
    // A part's own error says which part failed.
    let err = parsing_error(parser.parse_output("Confidence: A, Explanation: x\n\nJ: 1"));
    assert!(err.starts_with("part 2: no `K:`"), "{err}");

    let not_an_object = CombiningParser::new(confidence(), JsonParser::<Value>::new());
    let err = parsing_error(not_an_object.parse_output("Confidence: B, Explanation: y\n\n[1]"));
    assert!(
        err.starts_with("part 2: expected a JSON object, got [1]"),
        "{err}"
    );
}

#[test]
fn combining_parser_instructions_number_the_parts() {
    let parser = CombiningParser::new(
        RegexDictParser::new([("a", "A")]),
        RegexDictParser::new([("b", "B")]),
    );
    assert_eq!(
        parser.format_instructions(),
        "Your answer must have 2 parts, in this order, separated by a blank line. No part may contain a blank line itself.\n\n\
         Part 1:\nYour answer must contain these lines, each once:\nA: <value>\n\n\
         Part 2:\nYour answer must contain these lines, each once:\nB: <value>"
    );
}
