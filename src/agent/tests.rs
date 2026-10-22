//! Offline end-to-end tests for agents + `AgentExecutor`, driven by a scripted fake LLM.
use std::{
    error::Error,
    sync::{Arc, Mutex as StdMutex},
};

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::{
    agent::{AgentExecutor, ConversationalAgentBuilder, OpenAiToolAgentBuilder},
    chain::{Chain, ChainError},
    memory::SimpleMemory,
    prompt_args,
    schemas::{memory::BaseMemory, MessageType},
    test_utils::FakeLLM,
    tools::Tool,
};

/// Tool that records the inputs it receives and echoes a fixed answer (or fails).
#[derive(Clone)]
struct RecordingTool {
    name: String,
    answer: Result<String, String>,
    inputs: Arc<StdMutex<Vec<Value>>>,
}

impl RecordingTool {
    fn ok(name: &str, answer: &str) -> Self {
        Self {
            name: name.into(),
            answer: Ok(answer.into()),
            inputs: Default::default(),
        }
    }
    fn failing(name: &str, err: &str) -> Self {
        Self {
            name: name.into(),
            answer: Err(err.into()),
            inputs: Default::default(),
        }
    }
    fn inputs(&self) -> Vec<Value> {
        self.inputs.lock().unwrap().clone()
    }
}

#[async_trait]
impl Tool for RecordingTool {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn description(&self) -> String {
        format!("{} tool", self.name)
    }
    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        self.inputs.lock().unwrap().push(input);
        self.answer.clone().map_err(|e| e.into())
    }
}

fn action(tool: &str, input: &str) -> String {
    format!("```json\n{{\"action\": \"{tool}\", \"action_input\": \"{input}\"}}\n```")
}

fn final_answer(answer: &str) -> String {
    action("Final Answer", answer)
}

#[tokio::test]
async fn conversational_agent_uses_tool_then_answers() {
    let calc = RecordingTool::ok("Calculator", "4");
    let llm = FakeLLM::new([action("Calculator", "2+2"), final_answer("It is 4")]);
    let agent = ConversationalAgentBuilder::new()
        .tools(&[Arc::new(calc.clone())])
        .build(llm.clone())
        .unwrap();
    let memory = Arc::new(Mutex::new(SimpleMemory::new()));
    let executor = AgentExecutor::from_agent(agent).with_memory(memory.clone());

    let out = executor
        .invoke(prompt_args! {"input" => "what is 2+2?"})
        .await
        .unwrap();
    assert_eq!(out, "It is 4");
    assert_eq!(calc.inputs(), vec![json!("2+2")]);

    // Prompt contains the tool description and the user input
    let calls = llm.calls();
    assert_eq!(calls.len(), 2);
    let first_prompt: String = calls[0].iter().map(|m| m.content.clone()).collect();
    assert!(first_prompt.contains("> Calculator: Calculator tool"));
    assert!(first_prompt.contains("what is 2+2?"));
    // Second call carries the tool observation in the scratchpad
    let second_prompt: String = calls[1].iter().map(|m| m.content.clone()).collect();
    assert!(second_prompt.contains("TOOL RESPONSE"));
    assert!(second_prompt.contains('4'));

    // Default chain options (max_tokens = 1000) are pushed to the LLM
    let last_max_tokens = llm.options.lock().unwrap().last().unwrap().max_tokens;
    assert_eq!(last_max_tokens, Some(1000));

    let messages = memory.lock().await.messages().await.unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].message_type, MessageType::HumanMessage);
    assert_eq!(messages[0].content, "what is 2+2?");
    assert_eq!(messages[1].content, "It is 4");
}

#[tokio::test]
async fn tool_names_with_spaces_are_matched() {
    let tool = RecordingTool::ok("Web Search", "result");
    let llm = FakeLLM::new([action("Web Search", "rust"), final_answer("done")]);
    let agent = ConversationalAgentBuilder::new()
        .tools(&[Arc::new(tool.clone())])
        .build(llm)
        .unwrap();
    let out = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    assert_eq!(out, "done");
    assert_eq!(tool.inputs(), vec![json!("rust")]);
}

#[tokio::test]
async fn unknown_tool_is_an_error() {
    let llm = FakeLLM::new([action("Nope", "x")]);
    let agent = ConversationalAgentBuilder::new().build(llm).unwrap();
    let err = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ChainError::AgentError(m) if m.contains("Tool Nope not found")),
        "{err:?}"
    );
}

#[tokio::test]
async fn max_iterations_stops_the_loop() {
    let tool = RecordingTool::ok("Echo", "again");
    let llm = FakeLLM::new([
        action("Echo", "1"),
        action("Echo", "2"),
        action("Echo", "3"),
    ]);
    let agent = ConversationalAgentBuilder::new()
        .tools(&[Arc::new(tool.clone())])
        .build(llm)
        .unwrap();
    let out = AgentExecutor::from_agent(agent)
        .with_max_iterations(2)
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    assert_eq!(out, "Max iterations reached");
    assert_eq!(tool.inputs().len(), 2);
}

#[tokio::test]
async fn tool_errors_are_observations_unless_break_if_error() {
    let make = || {
        let tool = RecordingTool::failing("Broken", "boom");
        let llm = FakeLLM::new([action("Broken", "x"), final_answer("recovered")]);
        let agent = ConversationalAgentBuilder::new()
            .tools(&[Arc::new(tool)])
            .build(llm.clone())
            .unwrap();
        (agent, llm)
    };

    let (agent, llm) = make();
    let out = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    assert_eq!(out, "recovered");
    let second: String = llm.calls()[1].iter().map(|m| m.content.clone()).collect();
    assert!(second.contains("The tool return the following error: boom"));

    let (agent, _) = make();
    let err = AgentExecutor::from_agent(agent)
        .with_break_if_error(true)
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap_err();
    assert!(
        matches!(&err, ChainError::AgentError(m) if m.contains("boom")),
        "{err:?}"
    );
}

#[tokio::test]
async fn memory_history_is_sent_as_chat_history() {
    let llm = FakeLLM::new([final_answer("second answer")]);
    let agent = ConversationalAgentBuilder::new()
        .build(llm.clone())
        .unwrap();
    let mut mem = SimpleMemory::new();
    mem.add_user_message(&"earlier question").await.unwrap();
    mem.add_ai_message(&"earlier answer").await.unwrap();
    let memory = Arc::new(Mutex::new(mem));

    AgentExecutor::from_agent(agent)
        .with_memory(memory.clone())
        .invoke(prompt_args! {"input" => "new question"})
        .await
        .unwrap();

    let prompt = &llm.calls()[0];
    assert!(prompt
        .iter()
        .any(|m| m.content == "earlier question" && m.message_type == MessageType::HumanMessage));
    assert!(prompt
        .iter()
        .any(|m| m.content == "earlier answer" && m.message_type == MessageType::AIMessage));
    assert_eq!(memory.lock().await.messages().await.unwrap().len(), 4);
}

fn tool_calls_json(calls: &[(&str, &str, &str)]) -> String {
    json!(calls
        .iter()
        .map(|(id, name, args)| json!({
            "id": id, "type": "function",
            "function": {"name": name, "arguments": args}
        }))
        .collect::<Vec<_>>())
    .to_string()
}

#[tokio::test]
async fn openai_tools_agent_executes_parallel_tool_calls() {
    let weather = RecordingTool::ok("weather", "sunny");
    let time = RecordingTool::ok("time", "noon");
    let llm = FakeLLM::new([
        tool_calls_json(&[
            ("call_1", "weather", r#"{"input":"Lima"}"#),
            ("call_2", "time", "now"),
        ]),
        "Sunny at noon".to_string(),
    ]);
    // Tools collected at runtime in a Vec<Arc<dyn Tool>> work too (#323): &Vec derefs to a slice.
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(weather.clone()), Arc::new(time.clone())];
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&tools)
        .build(llm.clone())
        .unwrap();
    let memory = Arc::new(Mutex::new(SimpleMemory::new()));
    let out = AgentExecutor::from_agent(agent)
        .with_memory(memory.clone())
        .invoke(prompt_args! {"input" => "weather and time?"})
        .await
        .unwrap();

    assert_eq!(out, "Sunny at noon");
    // default Tool::parse_input unwraps {"input": ...}
    assert_eq!(weather.inputs(), vec![json!("Lima")]);
    assert_eq!(time.inputs(), vec![json!("now")]);

    // Tools were advertised as functions to the LLM
    let fn_names: Vec<String> = llm
        .options
        .lock()
        .unwrap()
        .iter()
        .flat_map(|o| o.functions.clone().unwrap_or_default())
        .map(|f| f.name)
        .collect();
    assert_eq!(fn_names, vec!["weather", "time"]);

    // Scratchpad on 2nd call: one AI message with tool_calls, then two tool messages
    let second = &llm.calls()[1];
    let ai_with_calls: Vec<_> = second.iter().filter(|m| m.tool_calls.is_some()).collect();
    assert_eq!(ai_with_calls.len(), 1);
    let tool_msgs: Vec<_> = second
        .iter()
        .filter(|m| m.message_type == MessageType::ToolMessage)
        .map(|m| (m.id.clone().unwrap(), m.content.clone()))
        .collect();
    assert_eq!(
        tool_msgs,
        vec![
            ("call_1".to_string(), "sunny".to_string()),
            ("call_2".to_string(), "noon".to_string())
        ]
    );

    // Memory: human, ai(tool_calls), tool, tool, ai(final)
    let types: Vec<_> = memory
        .lock()
        .await
        .messages()
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.message_type)
        .collect();
    assert_eq!(
        types,
        vec![
            MessageType::HumanMessage,
            MessageType::AIMessage,
            MessageType::ToolMessage,
            MessageType::ToolMessage,
            MessageType::AIMessage
        ]
    );
}

#[tokio::test]
async fn openai_tools_agent_plain_text_is_final() {
    let llm = FakeLLM::new(["hello"]);
    let agent = OpenAiToolAgentBuilder::new().build(llm).unwrap();
    let out = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "hi"})
        .await
        .unwrap();
    assert_eq!(out, "hello");
}

#[tokio::test]
async fn openai_tools_agent_uses_typed_tool_calls() {
    use crate::{language_models::GenerateResult, schemas::ToolCall};

    let weather = RecordingTool::ok("weather", "sunny");
    // `generation` is plain text: only the typed `tool_calls` can drive the action.
    let calls = vec![ToolCall::new("toolu_1", "weather", r#"{"input":"Lima"}"#)];
    let llm = FakeLLM::with_results([
        GenerateResult {
            generation: "Let me check".into(),
            tool_calls: calls.clone(),
            ..Default::default()
        },
        GenerateResult {
            generation: "Sunny in Lima".into(),
            ..Default::default()
        },
    ]);
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(weather.clone())];
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&tools)
        .build(llm.clone())
        .unwrap();
    let memory = Arc::new(Mutex::new(SimpleMemory::new()));
    let out = AgentExecutor::from_agent(agent)
        .with_memory(memory.clone())
        .invoke(prompt_args! {"input" => "weather?"})
        .await
        .unwrap();

    assert_eq!(out, "Sunny in Lima");
    assert_eq!(weather.inputs(), vec![json!("Lima")]);

    // The scratchpad replays the typed calls in the shared wire format.
    let second = &llm.calls()[1];
    let ai = second.iter().find(|m| m.tool_calls.is_some()).unwrap();
    assert_eq!(ai.parse_tool_calls(), calls);
    let tool_msg = second
        .iter()
        .find(|m| m.message_type == MessageType::ToolMessage)
        .unwrap();
    assert_eq!(
        (tool_msg.id.as_deref(), tool_msg.content.as_str()),
        (Some("toolu_1"), "sunny")
    );

    // ...and so does memory.
    let stored = memory.lock().await.messages().await.unwrap();
    assert_eq!(stored[1].parse_tool_calls(), calls);
}

#[tokio::test]
async fn openai_tools_agent_empty_or_invalid_json_list_is_final() {
    // A JSON array that is not tool calls must not be treated as an action.
    let llm = FakeLLM::new([r#"[1, 2, 3]"#, "[]"]);
    let agent = OpenAiToolAgentBuilder::new().build(llm).unwrap();
    let executor = AgentExecutor::from_agent(agent);
    let out = executor
        .invoke(prompt_args! {"input" => "list?"})
        .await
        .unwrap();
    assert_eq!(out, "[1, 2, 3]");
    let out = executor
        .invoke(prompt_args! {"input" => "empty?"})
        .await
        .unwrap();
    assert_eq!(out, "[]");
}

// ---- streaming + cancellation ----

use crate::{agent::AgentStreamEvent, language_models::GenerateResult, schemas::ToolCall};
use futures::StreamExt;
use tokio_util::sync::CancellationToken;

async fn collect_events(
    executor: &AgentExecutor<impl crate::agent::Agent>,
    input: &str,
) -> Vec<Result<AgentStreamEvent, ChainError>> {
    executor
        .stream_events(prompt_args! {"input" => input})
        .collect()
        .await
}

fn text(s: &str) -> AgentStreamEvent {
    AgentStreamEvent::Text(s.into())
}

#[tokio::test]
async fn stream_events_tools_agent_streams_tools_then_answer_tokens() {
    let weather = RecordingTool::ok("weather", "sunny");
    let llm = FakeLLM::with_results([
        GenerateResult::default().with_tool_calls(vec![ToolCall::new(
            "call_1",
            "weather",
            r#"{"input":"Lima"}"#,
        )]),
        GenerateResult {
            generation: "Sunny in Lima".into(),
            ..Default::default()
        },
    ]);
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(weather.clone())];
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&tools)
        .build(llm.clone())
        .unwrap();
    let memory = Arc::new(Mutex::new(SimpleMemory::new()));
    let executor = AgentExecutor::from_agent(agent).with_memory(memory.clone());

    let events: Vec<_> = collect_events(&executor, "weather?")
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(
        events,
        vec![
            AgentStreamEvent::ToolStart {
                tool: "weather".into(),
                input: r#"{"input":"Lima"}"#.into()
            },
            AgentStreamEvent::ToolEnd {
                tool: "weather".into(),
                output: "sunny".into()
            },
            // The final answer arrives token by token, not as one chunk.
            text("Sunny "),
            text("in "),
            text("Lima"),
            AgentStreamEvent::Finish {
                output: "Sunny in Lima".into()
            },
        ]
    );
    assert_eq!(weather.inputs(), vec![json!("Lima")]);

    // Memory is saved exactly like `call` does: human, ai(tool_calls), tool, ai(final).
    let types: Vec<_> = memory
        .lock()
        .await
        .messages()
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.message_type)
        .collect();
    assert_eq!(
        types,
        vec![
            MessageType::HumanMessage,
            MessageType::AIMessage,
            MessageType::ToolMessage,
            MessageType::AIMessage
        ]
    );
}

#[tokio::test]
async fn stream_events_default_plan_stream_works_for_any_agent() {
    // ConversationalAgent has no plan_stream override: the answer comes as one chunk.
    let calc = RecordingTool::ok("Calculator", "4");
    let llm = FakeLLM::new([action("Calculator", "2+2"), final_answer("It is 4")]);
    let agent = ConversationalAgentBuilder::new()
        .tools(&[Arc::new(calc.clone())])
        .build(llm)
        .unwrap();
    let events: Vec<_> = collect_events(&AgentExecutor::from_agent(agent), "2+2?")
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(
        events,
        vec![
            AgentStreamEvent::ToolStart {
                tool: "Calculator".into(),
                input: "2+2".into()
            },
            AgentStreamEvent::ToolEnd {
                tool: "Calculator".into(),
                output: "4".into()
            },
            text("It is 4"),
            AgentStreamEvent::Finish {
                output: "It is 4".into()
            },
        ]
    );
}

#[tokio::test]
async fn stream_events_tool_error_is_an_observation_event() {
    let broken = RecordingTool::failing("broken", "boom");
    let llm = FakeLLM::new([tool_calls_json(&[("c1", "broken", "x")]), "ok".into()]);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(broken)])
        .build(llm)
        .unwrap();
    let events: Vec<_> = collect_events(&AgentExecutor::from_agent(agent), "go")
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert!(
        events.contains(&AgentStreamEvent::ToolEnd {
            tool: "broken".into(),
            output: "The tool return the following error: boom".into()
        }),
        "{events:?}"
    );
    assert_eq!(
        events.last(),
        Some(&AgentStreamEvent::Finish {
            output: "ok".into()
        })
    );
}

#[tokio::test]
async fn stream_events_errors_end_the_stream_with_err() {
    // Unknown tool: ToolStart is never emitted, the stream ends with the error.
    let llm = FakeLLM::new([tool_calls_json(&[("c1", "nope", "x")])]);
    let agent = OpenAiToolAgentBuilder::new().build(llm).unwrap();
    let events = collect_events(&AgentExecutor::from_agent(agent), "go").await;
    assert_eq!(events.len(), 1, "{events:?}");
    let err = events[0].as_ref().unwrap_err().to_string();
    assert!(err.contains("Tool nope not found"), "{err}");

    // LLM failure (script exhausted) surfaces as an Err item too, not an empty stream.
    let agent = OpenAiToolAgentBuilder::new()
        .build(FakeLLM::new(Vec::<String>::new()))
        .unwrap();
    let events = collect_events(&AgentExecutor::from_agent(agent), "go").await;
    assert_eq!(events.len(), 1, "{events:?}");
    let err = events[0].as_ref().unwrap_err().to_string();
    assert!(err.contains("no scripted response left"), "{err}");
}

#[tokio::test]
async fn stream_events_max_iterations_finishes() {
    let tool = RecordingTool::ok("t", "again");
    let llm = FakeLLM::new([
        tool_calls_json(&[("c1", "t", "1")]),
        tool_calls_json(&[("c2", "t", "2")]),
    ]);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(tool)])
        .build(llm)
        .unwrap();
    let executor = AgentExecutor::from_agent(agent).with_max_iterations(2);
    let events: Vec<_> = collect_events(&executor, "loop")
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    let starts = events
        .iter()
        .filter(|e| matches!(e, AgentStreamEvent::ToolStart { .. }))
        .count();
    assert_eq!(starts, 2);
    assert_eq!(
        events.last(),
        Some(&AgentStreamEvent::Finish {
            output: "Max iterations reached".into()
        })
    );
}

#[tokio::test]
async fn cancellation_stops_a_running_agent_and_skips_memory() {
    // The 2nd LLM call never answers; cancelling must still end the run.
    let weather = RecordingTool::ok("weather", "sunny");
    let llm = FakeLLM::new([tool_calls_json(&[("c1", "weather", "Lima")])]).hang_when_exhausted();
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(weather.clone())])
        .build(llm.clone())
        .unwrap();
    let token = CancellationToken::new();
    let memory = Arc::new(Mutex::new(SimpleMemory::new()));
    let executor = AgentExecutor::from_agent(agent)
        .with_memory(memory.clone())
        .with_cancellation(token.clone());

    let mut events = executor.stream_events(prompt_args! {"input" => "weather?"});
    let mut seen = Vec::new();
    let drain = async {
        while let Some(event) = events.next().await {
            let done_with_tool = matches!(event, Ok(AgentStreamEvent::ToolEnd { .. }));
            seen.push(event);
            if done_with_tool {
                token.cancel();
            }
        }
    };
    tokio::time::timeout(std::time::Duration::from_secs(5), drain)
        .await
        .expect("cancel must end the stream");
    let err = seen.pop().unwrap().unwrap_err().to_string();
    assert!(err.contains("agent run cancelled"), "{err}");
    assert!(matches!(
        seen.as_slice(),
        [
            Ok(AgentStreamEvent::ToolStart { .. }),
            Ok(AgentStreamEvent::ToolEnd { .. })
        ]
    ));
    assert_eq!(
        llm.calls().len(),
        2,
        "second plan was in flight when cancelled"
    );
    assert!(memory.lock().await.messages().await.unwrap().is_empty());
}

#[tokio::test]
async fn cancellation_applies_to_call_too() {
    let llm = FakeLLM::new(Vec::<String>::new()).hang_when_exhausted();
    let agent = OpenAiToolAgentBuilder::new().build(llm).unwrap();
    let token = CancellationToken::new();
    let executor = AgentExecutor::from_agent(agent).with_cancellation(token.clone());

    let canceller = {
        let token = token.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            token.cancel();
        })
    };
    let err = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        executor.invoke(prompt_args! {"input" => "hi"}),
    )
    .await
    .expect("cancel must end the run")
    .unwrap_err();
    assert!(err.to_string().contains("agent run cancelled"), "{err}");
    canceller.await.unwrap();

    // An already-cancelled token stops the run before the LLM is called.
    let llm = FakeLLM::new(["never"]);
    let agent = OpenAiToolAgentBuilder::new().build(llm.clone()).unwrap();
    let executor = AgentExecutor::from_agent(agent).with_cancellation(token);
    assert!(executor
        .invoke(prompt_args! {"input" => "hi"})
        .await
        .is_err());
    assert!(llm.calls().is_empty());
}

#[tokio::test]
async fn dropping_the_stream_stops_the_run() {
    let tool = RecordingTool::ok("t", "x");
    let llm = FakeLLM::new([
        tool_calls_json(&[("c1", "t", "1")]),
        tool_calls_json(&[("c2", "t", "2")]),
        "done".into(),
    ]);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(tool.clone())])
        .build(llm.clone())
        .unwrap();
    let executor = AgentExecutor::from_agent(agent);
    {
        let mut events = executor.stream_events(prompt_args! {"input" => "go"});
        // Take the first tool start and walk away.
        let first = events.next().await.unwrap().unwrap();
        assert!(matches!(first, AgentStreamEvent::ToolStart { .. }));
    }
    // Lazily polled: once dropped, the run made no further LLM calls.
    assert_eq!(llm.calls().len(), 1);
    assert!(tool.inputs().len() <= 1);
}

#[tokio::test]
async fn call_and_stream_events_agree_on_the_answer() {
    let make = || {
        let llm = FakeLLM::new([tool_calls_json(&[("c1", "t", "q")]), "final answer".into()]);
        let agent = OpenAiToolAgentBuilder::new()
            .tools(&[Arc::new(RecordingTool::ok("t", "obs"))])
            .build(llm)
            .unwrap();
        AgentExecutor::from_agent(agent)
    };
    let via_call = make().invoke(prompt_args! {"input" => "q"}).await.unwrap();
    let streamed = collect_events(&make(), "q").await;
    let Some(Ok(AgentStreamEvent::Finish { output })) = streamed.last() else {
        panic!("{streamed:?}");
    };
    assert_eq!(&via_call, output);
    // Concatenated text chunks equal the final output.
    let joined: String = streamed
        .iter()
        .filter_map(|e| match e {
            Ok(AgentStreamEvent::Text(t)) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(joined, "final answer");
}

#[tokio::test]
async fn stream_events_never_leaks_text_tool_call_json_split_across_chunks() {
    // LLMs without native tool calling write the calls as JSON text; FakeLLM streams it
    // word by word, so the JSON arrives in many chunks and must still be held back.
    let spaced =
        r#"[{"id": "c1", "type": "function", "function": {"name": "t", "arguments": "q"}}]"#;
    let llm = FakeLLM::new([spaced, "the answer"]);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(RecordingTool::ok("t", "obs"))])
        .build(llm)
        .unwrap();
    let events: Vec<_> = collect_events(&AgentExecutor::from_agent(agent), "q")
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(
        events,
        vec![
            AgentStreamEvent::ToolStart {
                tool: "t".into(),
                input: "q".into()
            },
            AgentStreamEvent::ToolEnd {
                tool: "t".into(),
                output: "obs".into()
            },
            text("the "),
            text("answer"),
            AgentStreamEvent::Finish {
                output: "the answer".into()
            },
        ]
    );
}

#[tokio::test]
async fn stream_events_answer_starting_with_bracket_is_still_streamed() {
    // Held back as a possible tool call, but it is not one: the text must not be lost.
    let llm = FakeLLM::new(["[1, 2, 3] are the numbers"]);
    let agent = OpenAiToolAgentBuilder::new().build(llm).unwrap();
    let events: Vec<_> = collect_events(&AgentExecutor::from_agent(agent), "q")
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    let joined: String = events
        .iter()
        .filter_map(|e| match e {
            AgentStreamEvent::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(joined, "[1, 2, 3] are the numbers");
    assert_eq!(
        events.last(),
        Some(&AgentStreamEvent::Finish {
            output: "[1, 2, 3] are the numbers".into()
        })
    );
}
