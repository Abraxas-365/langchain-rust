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
async fn agents_send_input_images_and_save_them_to_memory() {
    // Tools agent: the images ride on the human message, also on the step after a tool.
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(RecordingTool::ok("Lookup", "an orange cat"))];
    let llm = FakeLLM::new([
        tool_calls_json(&[("call_1", "Lookup", "{}")]),
        "It's a cat".to_string(),
    ]);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&tools)
        .build(llm.clone())
        .unwrap();
    let memory = Arc::new(Mutex::new(SimpleMemory::new()));
    let out = AgentExecutor::from_agent(agent)
        .with_memory(memory.clone())
        .invoke(prompt_args! {
            "input" => "What is this?",
            "images" => ["https://example.com/cat.png", "iVBORw0KGgo"],
        })
        .await
        .unwrap();
    assert_eq!(out, "It's a cat");

    assert_eq!(llm.calls().len(), 2);
    for call in llm.calls() {
        let human: Vec<_> = call
            .iter()
            .filter(|m| m.message_type == MessageType::HumanMessage)
            .collect();
        assert_eq!(human.len(), 1);
        assert_eq!(human[0].content, "What is this?");
        let urls: Vec<_> = human[0]
            .images
            .iter()
            .flatten()
            .map(|i| i.image_url.as_str())
            .collect();
        assert_eq!(urls, ["https://example.com/cat.png", "iVBORw0KGgo"]);
    }
    let saved = memory.lock().await.messages().await.unwrap();
    assert_eq!(saved[0].message_type, MessageType::HumanMessage);
    assert_eq!(saved[0].content, "What is this?");
    assert_eq!(saved[0].images.as_ref().unwrap().len(), 2);

    // Conversational (ReAct-style) agent.
    let llm = FakeLLM::new([final_answer("A cat")]);
    let agent = ConversationalAgentBuilder::new()
        .build(llm.clone())
        .unwrap();
    AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "What is this?", "images" => "https://a/b.png"})
        .await
        .unwrap();
    let human = llm.calls()[0]
        .iter()
        .find(|m| m.message_type == MessageType::HumanMessage)
        .cloned()
        .unwrap();
    assert!(human.content.contains("What is this?"));
    assert_eq!(human.images.unwrap()[0].image_url, "https://a/b.png");
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

/// Drops `Usage` events (FakeLLM reports 1+1 tokens per call) for tests about the rest.
fn without_usage<E>(
    events: impl IntoIterator<Item = Result<AgentStreamEvent, E>>,
) -> Vec<Result<AgentStreamEvent, E>> {
    events
        .into_iter()
        .filter(|e| !matches!(e, Ok(AgentStreamEvent::Usage(_))))
        .collect()
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
    let events: Vec<_> =
        without_usage(collect_events(&AgentExecutor::from_agent(agent), "2+2?").await)
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
    let events = without_usage(collect_events(&AgentExecutor::from_agent(agent), "go").await);
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
        without_usage(seen).as_slice(),
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
        let first = loop {
            match events.next().await.unwrap().unwrap() {
                AgentStreamEvent::Usage(_) => continue,
                other => break other,
            }
        };
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
    let events: Vec<_> =
        without_usage(collect_events(&AgentExecutor::from_agent(agent), "q").await)
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

// ---- tracing spans ----

use crate::observability::testing::capture;

fn tools_agent_run() -> (RecordingTool, AgentExecutor<impl crate::agent::Agent>) {
    let weather = RecordingTool::ok("weather", "sunny");
    let llm = FakeLLM::with_results([
        GenerateResult::default().with_tool_calls(vec![ToolCall::new(
            "call_1",
            "weather",
            r#"{"input":"Lima"}"#,
        )]),
        GenerateResult {
            generation: "Sunny".into(),
            ..Default::default()
        },
    ]);
    let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(weather.clone())];
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&tools)
        .build(llm)
        .unwrap();
    (weather, AgentExecutor::from_agent(agent))
}

#[tokio::test]
async fn agent_run_emits_run_plan_and_tool_spans() {
    let (spans, _guard) = capture();
    let (_, executor) = tools_agent_run();
    executor
        .invoke(prompt_args! {"input" => "weather?"})
        .await
        .unwrap();

    let run = spans.named("agent.run");
    assert_eq!(run.len(), 1);
    assert_eq!(run[0].field("max_iterations"), Some("10"));
    assert_eq!(run[0].field("steps"), Some("1"));
    assert_eq!(run[0].field("error"), None);

    let plans = spans.named("agent.plan");
    let steps: Vec<_> = plans.iter().map(|p| p.field("step").unwrap()).collect();
    assert_eq!(steps, vec!["0", "1"], "one plan span per iteration");
    assert!(plans.iter().all(|p| p.parent == Some("agent.run")));

    let tool = &spans.named("tool.call")[0];
    assert_eq!(tool.field("tool"), Some("weather"));
    assert_eq!(tool.parent, Some("agent.run"));
    assert_eq!(tool.field("error"), None);

    // Each LLM call belongs to the step that made it (through the agent's LLMChain).
    let llm = spans.named("llm.generate");
    assert_eq!(llm.len(), 2);
    assert!(
        llm.iter().all(|s| s.parent == Some("chain.call")),
        "{llm:?}"
    );
    let chains = spans.named("chain.call");
    assert!(
        chains.iter().all(|s| s.parent == Some("agent.plan")),
        "{chains:?}"
    );
}

#[tokio::test]
async fn agent_spans_are_the_same_when_streaming() {
    let (spans, _guard) = capture();
    let (_, executor) = tools_agent_run();
    let events = collect_events(&executor, "weather?").await;
    assert!(events.iter().all(Result::is_ok));

    assert_eq!(spans.named("agent.run")[0].field("steps"), Some("1"));
    assert_eq!(spans.named("agent.plan").len(), 2);
    assert_eq!(spans.named("tool.call")[0].parent, Some("agent.run"));
    let llm = spans.named("llm.generate");
    assert_eq!(llm.len(), 2);
    assert!(llm.iter().all(|s| s.field("stream") == Some("true")));
    assert!(
        llm.iter().all(|s| s.parent == Some("agent.plan")),
        "{llm:?}"
    );
}

#[tokio::test]
async fn failing_tool_records_error_on_tool_span() {
    let (spans, _guard) = capture();
    let llm = FakeLLM::new([action("Broken", "x"), final_answer("recovered")]);
    let agent = ConversationalAgentBuilder::new()
        .tools(&[Arc::new(RecordingTool::failing("Broken", "boom"))])
        .build(llm)
        .unwrap();
    let out = AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    assert_eq!(out, "recovered");

    let tool = &spans.named("tool.call")[0];
    assert_eq!(tool.field("tool"), Some("Broken"));
    assert_eq!(tool.field("error"), Some("boom"));
    // The tool error became an observation, so the run itself succeeded.
    assert_eq!(spans.named("agent.run")[0].field("error"), None);
}

#[tokio::test]
async fn failed_run_records_error_on_run_span() {
    let (spans, _guard) = capture();
    let llm = FakeLLM::new([action("Nope", "x")]);
    let agent = ConversationalAgentBuilder::new().build(llm).unwrap();
    AgentExecutor::from_agent(agent)
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap_err();
    let run = &spans.named("agent.run")[0];
    assert!(run.field("error").unwrap().contains("Tool Nope not found"));
}

#[tokio::test]
async fn max_iterations_records_steps() {
    let (spans, _guard) = capture();
    let llm = FakeLLM::new([action("Echo", "1"), action("Echo", "2")]);
    let agent = ConversationalAgentBuilder::new()
        .tools(&[Arc::new(RecordingTool::ok("Echo", "again"))])
        .build(llm)
        .unwrap();
    AgentExecutor::from_agent(agent)
        .with_max_iterations(2)
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    assert_eq!(spans.named("agent.run")[0].field("steps"), Some("2"));
    assert_eq!(spans.named("tool.call").len(), 2);
}

// ---- token usage ----

use crate::language_models::TokenUsage;

fn with_usage(result: GenerateResult, prompt: u32, completion: u32) -> GenerateResult {
    GenerateResult {
        tokens: Some(TokenUsage::new(prompt, completion)),
        ..result
    }
}

/// Tools agent over a 2-call run: a tool call (10+2 tokens), then the answer (30+5).
fn usage_tools_run() -> AgentExecutor<impl crate::agent::Agent> {
    let llm = FakeLLM::with_results([
        with_usage(
            GenerateResult::default().with_tool_calls(vec![ToolCall::new("c1", "t", "q")]),
            10,
            2,
        ),
        with_usage(
            GenerateResult {
                generation: "done".into(),
                ..Default::default()
            },
            30,
            5,
        ),
    ]);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(RecordingTool::ok("t", "obs"))])
        .build(llm)
        .unwrap();
    AgentExecutor::from_agent(agent)
}

#[tokio::test]
async fn call_sums_token_usage_over_every_plan() {
    let result = usage_tools_run()
        .call(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    assert_eq!(result.generation, "done");
    assert_eq!(result.tokens, Some(TokenUsage::new(40, 7)));
}

#[tokio::test]
async fn run_span_records_total_tokens() {
    let (spans, _guard) = capture();
    usage_tools_run()
        .invoke(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    assert_eq!(
        spans.named("agent.run")[0].field("total_tokens"),
        Some("47")
    );
}

#[tokio::test]
async fn streaming_run_span_records_total_tokens_too() {
    let (spans, _guard) = capture();
    let events = collect_events(&usage_tools_run(), "q").await;
    assert!(events.iter().all(Result::is_ok), "{events:?}");
    assert_eq!(
        spans.named("agent.run")[0].field("total_tokens"),
        Some("47")
    );
}

#[tokio::test]
async fn conversational_agent_reports_token_usage() {
    let llm = FakeLLM::with_results([
        with_usage(
            GenerateResult {
                generation: action("Calculator", "2+2"),
                ..Default::default()
            },
            7,
            1,
        ),
        with_usage(
            GenerateResult {
                generation: final_answer("4"),
                ..Default::default()
            },
            9,
            3,
        ),
    ]);
    let agent = ConversationalAgentBuilder::new()
        .tools(&[Arc::new(RecordingTool::ok("Calculator", "4"))])
        .build(llm)
        .unwrap();
    let executor = AgentExecutor::from_agent(agent);
    let result = executor
        .call(prompt_args! {"input" => "2+2"})
        .await
        .unwrap();
    assert_eq!(result.tokens, Some(TokenUsage::new(16, 4)));
}

#[tokio::test]
async fn stream_events_reports_usage_per_plan_before_its_actions() {
    let events: Vec<_> = collect_events(&usage_tools_run(), "q")
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(
        events,
        vec![
            AgentStreamEvent::Usage(TokenUsage::new(10, 2)),
            AgentStreamEvent::ToolStart {
                tool: "t".into(),
                input: "q".into()
            },
            AgentStreamEvent::ToolEnd {
                tool: "t".into(),
                output: "obs".into()
            },
            text("done"),
            AgentStreamEvent::Usage(TokenUsage::new(30, 5)),
            AgentStreamEvent::Finish {
                output: "done".into()
            },
        ]
    );
}

#[tokio::test]
async fn default_plan_stream_forwards_usage() {
    // ConversationalAgent uses the trait's default plan_stream.
    let llm = FakeLLM::with_results([with_usage(
        GenerateResult {
            generation: final_answer("hi"),
            ..Default::default()
        },
        4,
        2,
    )]);
    let agent = ConversationalAgentBuilder::new().build(llm).unwrap();
    let events: Vec<_> = collect_events(&AgentExecutor::from_agent(agent), "q")
        .await
        .into_iter()
        .map(Result::unwrap)
        .collect();
    assert_eq!(
        events,
        vec![
            text("hi"),
            AgentStreamEvent::Usage(TokenUsage::new(4, 2)),
            AgentStreamEvent::Finish {
                output: "hi".into()
            },
        ]
    );
}

#[tokio::test]
async fn missing_usage_stays_none() {
    // An LLM that reports no usage: no Usage events and `tokens` is None, not zero.
    let make = || {
        let llm = FakeLLM::with_results([GenerateResult {
            generation: "plain".into(),
            ..Default::default()
        }]);
        AgentExecutor::from_agent(OpenAiToolAgentBuilder::new().build(llm).unwrap())
    };
    let result = make().call(prompt_args! {"input" => "q"}).await.unwrap();
    assert_eq!(result.tokens, None);
    let events = collect_events(&make(), "q").await;
    assert!(
        !events
            .iter()
            .any(|e| matches!(e, Ok(AgentStreamEvent::Usage(_)))),
        "{events:?}"
    );
}

#[tokio::test]
async fn max_iterations_still_reports_usage() {
    let llm = FakeLLM::with_results([
        with_usage(
            GenerateResult::default().with_tool_calls(vec![ToolCall::new("c1", "t", "1")]),
            5,
            1,
        ),
        with_usage(
            GenerateResult::default().with_tool_calls(vec![ToolCall::new("c2", "t", "2")]),
            6,
            1,
        ),
    ]);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(RecordingTool::ok("t", "again"))])
        .build(llm)
        .unwrap();
    let result = AgentExecutor::from_agent(agent)
        .with_max_iterations(2)
        .call(prompt_args! {"input" => "q"})
        .await
        .unwrap();
    assert_eq!(result.generation, "Max iterations reached");
    assert_eq!(result.tokens, Some(TokenUsage::new(11, 2)));
}

/// Live: `OLLAMA_MODEL=qwen3:14b cargo test --lib agent_usage_live -- --ignored`
#[tokio::test]
#[ignore]
async fn agent_usage_live_ollama() {
    use crate::{
        language_models::options::CallOptions,
        llm::{ollama::openai::OllamaConfig, openai::OpenAI},
    };
    let model = std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:14b".into());
    let make = || {
        let llm = OpenAI::new(OllamaConfig::default())
            .with_model(&model)
            .with_options(CallOptions::new().with_stream_usage(true));
        let agent = OpenAiToolAgentBuilder::new()
            .tools(&[Arc::new(RecordingTool::ok("weather", "sunny, 24C"))])
            .build(llm)
            .unwrap();
        AgentExecutor::from_agent(agent)
    };
    let input = "Use the weather tool to check Lima, then answer in one sentence.";

    let result = make().call(prompt_args! {"input" => input}).await.unwrap();
    let total = result.tokens.expect("usage from a non-streaming run");
    println!("call: {:?} -> {}", total, result.generation);
    assert!(total.prompt_tokens > 0 && total.completion_tokens > 0);

    let events = collect_events(&make(), input).await;
    let usages: Vec<_> = events
        .iter()
        .filter_map(|e| match e {
            Ok(AgentStreamEvent::Usage(u)) => Some(u.clone()),
            _ => None,
        })
        .collect();
    println!("stream usages: {usages:?}");
    assert!(!usages.is_empty(), "{events:?}");
}
