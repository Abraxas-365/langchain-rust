//! Offline tests for `ReActAgent` and the executor features it relies on (parser error
//! handler, intermediate steps), driven by a scripted fake LLM.
use std::sync::Arc;

use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::{
    agent::{
        tests::RecordingTool, AgentExecutor, AgentStreamEvent, ConversationalAgentBuilder,
        ParserErrorHandler, ReActAgent, ReActAgentBuilder, INTERMEDIATE_STEPS_KEY,
        PARSER_ERROR_TOOL,
    },
    chain::{options::ChainCallOptions, Chain, ChainError},
    memory::SimpleMemory,
    prompt_args,
    schemas::{memory::BaseMemory, AgentAction},
    test_utils::FakeLLM,
    tools::Tool,
};

/// What the model is told when its turn has no `Action:`.
const MISSING: &str = "Invalid Format: Missing 'Action:' after 'Thought:'. To use a tool, write 'Action:' and 'Action Input:'; if you know the answer, write 'Final Answer:' followed by it.";

fn react(tools: &[RecordingTool], llm: &FakeLLM) -> ReActAgent {
    let tools: Vec<Arc<dyn Tool>> = tools
        .iter()
        .map(|t| Arc::new(t.clone()) as Arc<dyn Tool>)
        .collect();
    ReActAgentBuilder::new()
        .tools(&tools)
        .build(llm.clone())
        .unwrap()
}

/// The single (human) message of the `i`-th model call.
fn prompt(llm: &FakeLLM, i: usize) -> String {
    let calls = llm.calls();
    assert_eq!(calls[i].len(), 1, "the ReAct prompt is one message");
    calls[i][0].content.clone()
}

const EXPECTED_PROMPT: &str =
    "Answer the following questions as best you can. You have access to the following tools:

search: search tool
calc: calc tool

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [search, calc]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: how old is the captain?
Thought:";

#[tokio::test]
async fn react_agent_runs_tools_then_answers() {
    let search = RecordingTool::ok("search", "the captain is 40");
    let calc = RecordingTool::ok("calc", "80");
    let turn1 = " I should look it up.\nAction: search\nAction Input: captain age";
    let turn2 = " Now double it.\nAction: calc\nAction Input: 40 * 2";
    let llm = FakeLLM::new([
        turn1,
        turn2,
        " I now know the final answer\nFinal Answer: 80",
    ]);
    let executor = AgentExecutor::from_agent(react(&[search.clone(), calc.clone()], &llm));

    let out = executor
        .invoke(prompt_args! {"input" => "how old is the captain?"})
        .await
        .unwrap();
    assert_eq!(out, "80");
    assert_eq!(search.inputs(), vec![json!("captain age")]);
    assert_eq!(calc.inputs(), vec![json!("40 * 2")]);

    assert_eq!(llm.calls().len(), 3);
    assert_eq!(prompt(&llm, 0), EXPECTED_PROMPT);
    let scratch1 = format!("{turn1}\nObservation: the captain is 40\nThought: ");
    assert_eq!(prompt(&llm, 1), format!("{EXPECTED_PROMPT}{scratch1}"));
    let scratch2 = format!("{scratch1}{turn2}\nObservation: 80\nThought: ");
    assert_eq!(prompt(&llm, 2), format!("{EXPECTED_PROMPT}{scratch2}"));
}

#[tokio::test]
async fn react_agent_stops_at_observation_and_keeps_user_options() {
    let llm = FakeLLM::new(["Final Answer: ok"]);
    let agent = ReActAgentBuilder::new()
        .options(
            ChainCallOptions::new()
                .with_max_tokens(77)
                .with_stop_words(vec!["STOP".into(), "\nObservation:".into()]),
        )
        .build(llm.clone())
        .unwrap();
    AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    let options = llm.options.lock().unwrap().last().cloned().unwrap();
    assert_eq!(options.max_tokens, Some(77));
    assert_eq!(
        options.stop_words,
        Some(vec![
            "STOP".to_string(),
            "\nObservation:".to_string(),
            "\n\tObservation:".to_string(),
        ]),
        "user stop words kept, ours added once"
    );

    let llm = FakeLLM::new(["Final Answer: ok"]);
    let agent = ReActAgentBuilder::new()
        .stop_at_observation(false)
        .build(llm.clone())
        .unwrap();
    AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    let options = llm.options.lock().unwrap().last().cloned().unwrap();
    assert_eq!(options.stop_words, None);
}

#[tokio::test]
async fn react_agent_custom_prompt_parts() {
    let llm = FakeLLM::new(["Final Answer: ok"]);
    let tool: Arc<dyn Tool> = Arc::new(RecordingTool::ok("calc", "1"));
    let agent = ReActAgentBuilder::new()
        .tools(&[tool])
        .prefix("Tools: {{tool_names}} / {{tool_descriptions}}")
        .format_instructions("Be brief.")
        .suffix("Q={{input}} S={{agent_scratchpad}}")
        .build(llm.clone())
        .unwrap();
    AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "hi"})
        .await
        .unwrap();
    assert_eq!(
        prompt(&llm, 0),
        "Tools: calc / calc: calc tool\n\nBe brief.\n\nQ=hi S="
    );
}

#[test]
fn react_builder_rejects_a_prompt_without_input_or_scratchpad() {
    for suffix in ["Question: {{input}}", "Thought:{{agent_scratchpad}}"] {
        let err = ReActAgentBuilder::new()
            .suffix(suffix)
            .build(FakeLLM::new(Vec::<String>::new()))
            .err()
            .expect("missing variable");
        assert!(
            matches!(err, crate::agent::AgentError::MissingInputVariable(_)),
            "{err:?}"
        );
    }
}

#[test]
fn react_scratchpad_format() {
    let step = |log: &str, obs: &str| {
        (
            AgentAction {
                tool: "t".into(),
                tool_input: "i".into(),
                log: log.into(),
            },
            obs.to_string(),
        )
    };
    assert_eq!(ReActAgent::construct_scratchpad(&[]), "");
    assert_eq!(
        ReActAgent::construct_scratchpad(&[step(" a\nAction: t", "x"), step(" b", "y")]),
        " a\nAction: t\nObservation: x\nThought:  b\nObservation: y\nThought: "
    );
}

#[tokio::test]
async fn react_agent_shows_memory_as_previous_conversation() {
    let memory = Arc::new(Mutex::new(SimpleMemory::new()));
    let llm = FakeLLM::new(["Final Answer: Hi Luis", "Final Answer: Luis"]);
    let executor = AgentExecutor::from_agent(react(&[], &llm)).with_memory(memory.clone());

    executor
        .invoke(prompt_args! {"input" => "I am Luis"})
        .await
        .unwrap();
    assert!(
        !prompt(&llm, 0).contains("Previous conversation"),
        "no history section on the first turn"
    );
    executor
        .invoke(prompt_args! {"input" => "who am I?"})
        .await
        .unwrap();
    assert!(
        prompt(&llm, 1).contains(
            "Begin!\n\nPrevious conversation:\nHumanMessage: I am Luis\nAIMessage: Hi Luis\n\nQuestion: who am I?\nThought:"
        ),
        "{}",
        prompt(&llm, 1)
    );
    assert_eq!(memory.lock().await.messages().await.unwrap().len(), 4);
}

#[tokio::test]
async fn parse_errors_fail_the_run_without_a_handler() {
    let llm = FakeLLM::new(["I think it's 42."]);
    let err = AgentExecutor::from_agent(react(&[], &llm))
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ChainError::AgentError(m)
            if m.contains("Missing 'Action:'") && m.contains("I think it's 42.")),
        "{err:?}"
    );
}

#[tokio::test]
async fn parser_error_handler_sends_the_error_back_and_retries() {
    let search = RecordingTool::ok("search", "found");
    let bad = " I'll just guess.";
    let llm = FakeLLM::new([
        bad,
        " Let me search.\nAction: search\nAction Input: x",
        "Final Answer: found it",
    ]);
    let executor = AgentExecutor::from_agent(react(std::slice::from_ref(&search), &llm))
        .with_parser_error_handler(ParserErrorHandler::new())
        .with_return_intermediate_steps(true);

    let out = executor
        .execute(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    assert_eq!(out["output"], json!("found it"));
    assert_eq!(search.inputs(), vec![json!("x")]);
    // The model sees its bad turn and what was wrong with it.
    assert!(prompt(&llm, 1).ends_with(&format!("Thought:{bad}\nObservation: {MISSING}\nThought: ")));
    let steps = out[INTERMEDIATE_STEPS_KEY].as_array().unwrap();
    assert_eq!(steps.len(), 2);
    assert_eq!(steps[0]["tool"], json!(PARSER_ERROR_TOOL));
    assert_eq!(steps[0]["log"], json!(bad));
    assert_eq!(steps[1]["tool"], json!("search"));
}

#[tokio::test]
async fn parser_error_handler_formatter_shapes_the_observation() {
    let llm = FakeLLM::new(["oops", "Final Answer: ok"]);
    let executor = AgentExecutor::from_agent(react(&[], &llm)).with_parser_error_handler(
        ParserErrorHandler::new().with_formatter(|err| format!("[{err}] try again")),
    );
    assert_eq!(
        executor
            .invoke(prompt_args! {"input" => "q"})
            .await
            .unwrap(),
        "ok"
    );
    assert!(prompt(&llm, 1).ends_with(&format!(
        "Thought:oops\nObservation: [{MISSING}] try again\nThought: "
    )));
}

#[tokio::test]
async fn parser_error_retries_count_toward_max_iterations() {
    let llm = FakeLLM::new(["bad 1", "bad 2", "bad 3"]);
    let executor = AgentExecutor::from_agent(react(&[], &llm))
        .with_max_iterations(2)
        .with_parser_error_handler(ParserErrorHandler::new())
        .with_return_intermediate_steps(true);
    let out = executor
        .execute(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    assert_eq!(out["output"], json!("Max iterations reached"));
    assert_eq!(llm.calls().len(), 2);
    assert_eq!(out[INTERMEDIATE_STEPS_KEY].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn unknown_tool_is_an_observation_with_a_handler() {
    let search = RecordingTool::ok("search", "found");
    let calc = RecordingTool::ok("calc", "2");
    let llm = FakeLLM::new(["Action: browse\nAction Input: x", "Final Answer: gave up"]);
    let executor = AgentExecutor::from_agent(react(&[search.clone(), calc.clone()], &llm))
        .with_parser_error_handler(ParserErrorHandler::new());
    assert_eq!(
        executor
            .invoke(prompt_args! {"input" => "q"})
            .await
            .unwrap(),
        "gave up"
    );
    assert!(prompt(&llm, 1).ends_with(
        "Observation: browse is not a valid tool, try one of [search, calc].\nThought: "
    ));
    assert!(search.inputs().is_empty() && calc.inputs().is_empty());

    // Without a handler it is still an error, as before.
    let llm = FakeLLM::new(["Action: browse\nAction Input: x"]);
    let err = AgentExecutor::from_agent(react(&[search], &llm))
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ChainError::AgentError(m) if m.contains("Tool browse not found")),
        "{err:?}"
    );
}

#[tokio::test]
async fn intermediate_steps_are_returned_only_when_asked() {
    let tool = RecordingTool::failing("calc", "division by zero");
    let turn = " Divide.\nAction: calc\nAction Input: 1/0";
    let script = [turn, "Final Answer: undefined"];

    let llm = FakeLLM::new(script);
    let executor = AgentExecutor::from_agent(react(std::slice::from_ref(&tool), &llm))
        .with_return_intermediate_steps(true);
    assert!(executor
        .get_output_keys()
        .contains(&INTERMEDIATE_STEPS_KEY.to_string()));
    let out = executor
        .execute(prompt_args! {"input" => "1/0?"})
        .await
        .unwrap();
    assert_eq!(out["output"], json!("undefined"));
    assert_eq!(out["generate_result"]["generation"], json!("undefined"));
    assert_eq!(
        out[INTERMEDIATE_STEPS_KEY],
        json!([{
            "tool": "calc",
            "tool_input": "1/0",
            "log": turn,
            "observation": "The tool return the following error: division by zero",
        }])
    );

    let llm = FakeLLM::new(script);
    let executor = AgentExecutor::from_agent(react(&[tool], &llm));
    assert_eq!(
        executor.get_output_keys(),
        vec!["output".to_string(), "generate_result".to_string()]
    );
    let out = executor
        .execute(prompt_args! {"input" => "1/0?"})
        .await
        .unwrap();
    assert_eq!(out.len(), 2, "{out:?}");
    assert!(!out.contains_key(INTERMEDIATE_STEPS_KEY));
}

#[tokio::test]
async fn intermediate_steps_work_for_other_agents_and_on_max_iterations() {
    let echo = RecordingTool::ok("Echo", "again");
    let json_action = "```json\n{\"action\": \"Echo\", \"action_input\": \"1\"}\n```";
    let llm = FakeLLM::new([json_action, json_action, json_action]);
    let agent = ConversationalAgentBuilder::new()
        .tools(&[Arc::new(echo)])
        .build(llm)
        .unwrap();
    let out = AgentExecutor::from_agent(agent)
        .with_max_iterations(2)
        .with_return_intermediate_steps(true)
        .execute(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    assert_eq!(out["output"], json!("Max iterations reached"));
    let steps = out[INTERMEDIATE_STEPS_KEY].as_array().unwrap();
    assert_eq!(steps.len(), 2);
    for step in steps {
        assert_eq!(step["tool"], json!("Echo"));
        assert_eq!(step["tool_input"], json!("1"));
        assert_eq!(step["observation"], json!("again"));
    }
}

#[tokio::test]
async fn placeholders_in_values_are_not_substituted_again() {
    let echo = RecordingTool::ok("echo", "out {{input}} {{chat_history}}");
    let llm = FakeLLM::new(["Action: echo\nAction Input: x", "Final Answer: ok"]);
    let executor = AgentExecutor::from_agent(react(&[echo], &llm));
    executor
        .invoke(prompt_args! {"input" => "what is {{agent_scratchpad}} {{tool_names}}?"})
        .await
        .unwrap();
    let second = prompt(&llm, 1);
    assert!(
        second.ends_with(
            "Question: what is {{agent_scratchpad}} {{tool_names}}?\nThought:Action: echo\nAction Input: x\nObservation: out {{input}} {{chat_history}}\nThought: "
        ),
        "{second}"
    );
}

#[tokio::test]
async fn react_agent_streams_and_retries_parse_errors() {
    use futures::StreamExt;
    let search = RecordingTool::ok("search", "found");
    let llm = FakeLLM::new([
        "garbage",
        "Action: browse\nAction Input: y",
        "Action: search\nAction Input: x",
        "Final Answer: done",
    ]);
    let executor = AgentExecutor::from_agent(react(&[search], &llm))
        .with_parser_error_handler(ParserErrorHandler::new());
    let events: Vec<AgentStreamEvent> = executor
        .stream_events(prompt_args! {"input" => "q"})
        .map(|e| e.unwrap())
        .filter(|e| futures::future::ready(!matches!(e, AgentStreamEvent::Usage(_))))
        .collect()
        .await;
    assert_eq!(
        events,
        vec![
            AgentStreamEvent::ToolStart {
                tool: "browse".into(),
                input: "y".into()
            },
            AgentStreamEvent::ToolEnd {
                tool: "browse".into(),
                output: "browse is not a valid tool, try one of [search].".into()
            },
            AgentStreamEvent::ToolStart {
                tool: "search".into(),
                input: "x".into()
            },
            AgentStreamEvent::ToolEnd {
                tool: "search".into(),
                output: "found".into()
            },
            AgentStreamEvent::Text("done".into()),
            AgentStreamEvent::Finish {
                output: "done".into()
            },
        ]
    );
}

#[tokio::test]
async fn react_steps_json_is_valid_for_quoted_text() {
    let llm = FakeLLM::new([
        "Action: say\nAction Input: \"hi\" \\ there",
        "Final Answer: ok",
    ]);
    let say = RecordingTool::ok("say", "said \"hi\"");
    let out = AgentExecutor::from_agent(react(std::slice::from_ref(&say), &llm))
        .with_return_intermediate_steps(true)
        .execute(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    let step: &Value = &out[INTERMEDIATE_STEPS_KEY][0];
    assert_eq!(step["tool_input"], json!("hi\" \\ there"));
    assert_eq!(step["observation"], json!("said \"hi\""));
}

/// Tool for the live test: multiplies the two integers in its input.
#[derive(Clone)]
struct Multiply;

#[async_trait::async_trait]
impl Tool for Multiply {
    fn name(&self) -> String {
        "multiply".into()
    }
    fn description(&self) -> String {
        "Multiplies two integers. Input: the two numbers separated by a comma, e.g. `3, 4`.".into()
    }
    async fn run(&self, input: Value) -> Result<String, Box<dyn std::error::Error>> {
        let text = input.as_str().unwrap_or_default().to_string();
        let nums: Vec<i64> = text
            .split(|c: char| !c.is_ascii_digit() && c != '-')
            .filter_map(|n| n.parse().ok())
            .collect();
        match nums.as_slice() {
            [a, b] => Ok((a * b).to_string()),
            _ => Err(format!("expected two integers, got `{text}`").into()),
        }
    }
}

/// Live: `OLLAMA_MODEL=qwen3:14b cargo test --lib react_agent_live -- --ignored --nocapture`
#[tokio::test]
#[ignore]
async fn react_agent_live_ollama() {
    use crate::llm::{ollama::openai::OllamaConfig, openai::OpenAI};
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:14b".into());
    let llm = OpenAI::new(OllamaConfig::default()).with_model(&model);
    let agent = ReActAgentBuilder::new()
        .tools(&[Arc::new(Multiply)])
        .build(llm)
        .unwrap();
    let executor = AgentExecutor::from_agent(agent)
        .with_parser_error_handler(ParserErrorHandler::new())
        .with_return_intermediate_steps(true);
    let out = executor
        .execute(prompt_args! {
            "input" => "Use the multiply tool to compute 1234 times 5678. Reply with just the number."
        })
        .await
        .unwrap();
    println!(
        "{}",
        serde_json::to_string_pretty(&out[INTERMEDIATE_STEPS_KEY]).unwrap()
    );
    println!("answer: {}", out["output"]);
    let steps = out[INTERMEDIATE_STEPS_KEY].as_array().unwrap();
    assert!(
        steps.iter().any(|s| s["tool"] == json!("multiply")),
        "the model used the tool"
    );
    assert!(
        out["output"]
            .as_str()
            .unwrap()
            .replace(',', "")
            .contains("7006652"),
        "{}",
        out["output"]
    );
}
