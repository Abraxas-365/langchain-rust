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
