use std::sync::{Arc, Mutex};

use serde::Deserialize;
use serde_json::{json, Value};
use tokio::sync::Mutex as AsyncMutex;

use super::FunctionTool;
use crate::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::Chain,
    memory::SimpleMemory,
    prompt_args,
    schemas::MessageType,
    test_utils::FakeLLM,
    tools::Tool,
};

#[derive(Debug, Deserialize, PartialEq)]
struct AddInput {
    a: f64,
    b: f64,
}

fn add_schema() -> Value {
    json!({
        "type": "object",
        "properties": { "a": {"type": "number"}, "b": {"type": "number"} },
        "required": ["a", "b"]
    })
}

fn adder() -> FunctionTool<AddInput, impl Fn(AddInput) -> std::future::Ready<Result<String, String>>>
{
    FunctionTool::new(
        "adder",
        "Adds two numbers together",
        add_schema(),
        |input: AddInput| {
            std::future::ready(Ok(format!(
                "The sum of {} and {} is {}",
                input.a,
                input.b,
                input.a + input.b
            )))
        },
    )
}

#[tokio::test]
async fn typed_tool_exposes_its_metadata_and_runs() {
    let tool = adder();
    assert_eq!(tool.name(), "adder");
    assert_eq!(tool.description(), "Adds two numbers together");
    assert_eq!(tool.parameters(), add_schema());

    let out = tool.call(r#"{"a": 2, "b": 3.5}"#).await.unwrap();
    assert_eq!(out, "The sum of 2 and 3.5 is 5.5");
    let out = tool.run(json!({"a": 1, "b": 1})).await.unwrap();
    assert_eq!(out, "The sum of 1 and 1 is 2");
}

#[tokio::test]
async fn typed_tool_accepts_wrapped_and_double_encoded_arguments() {
    let tool = adder();
    let wrapped = tool.call(r#"{"input": {"a": 1, "b": 2}}"#).await.unwrap();
    let encoded = tool.call(r#""{\"a\": 1, \"b\": 2}""#).await.unwrap();
    assert_eq!(wrapped, "The sum of 1 and 2 is 3");
    assert_eq!(encoded, wrapped);
}

#[tokio::test]
async fn bad_input_is_an_error_naming_the_schema_and_skips_the_function() {
    let calls = Arc::new(Mutex::new(0));
    let seen = calls.clone();
    let tool = FunctionTool::new("adder", "adds", add_schema(), move |input: AddInput| {
        *seen.lock().unwrap() += 1;
        async move { Ok::<_, String>(input.a + input.b) }
    });

    for input in [r#"{"a": 1}"#, "two plus two", r#"{"a": "x", "b": 1}"#] {
        let err = tool.call(input).await.unwrap_err().to_string();
        assert!(err.contains("invalid input for tool `adder`"), "{err}");
        assert!(err.contains(r#""required":["a","b"]"#), "{err}");
    }
    assert!(tool
        .call(r#"{"a": 1}"#)
        .await
        .unwrap_err()
        .to_string()
        .contains("missing field `b`"));
    assert_eq!(*calls.lock().unwrap(), 0);
}

#[tokio::test]
async fn function_errors_and_outputs_are_passed_through() {
    let tool = FunctionTool::new(
        "div",
        "divides",
        add_schema(),
        |input: AddInput| async move {
            if input.b == 0.0 {
                Err("division by zero")
            } else {
                Ok(input.a / input.b)
            }
        },
    );
    // Any `ToString` output works; here an f64.
    assert_eq!(tool.call(r#"{"a": 1, "b": 4}"#).await.unwrap(), "0.25");
    let err = tool.call(r#"{"a": 1, "b": 0}"#).await.unwrap_err();
    assert_eq!(err.to_string(), "division by zero");
}

#[tokio::test]
async fn text_tool_takes_the_input_as_a_string() {
    let tool = FunctionTool::text("shout", "Upper-cases the input", |s: String| async move {
        Ok::<_, String>(s.to_uppercase())
    });
    assert_eq!(
        tool.parameters(),
        json!({
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Upper-cases the input"}
            },
            "required": ["input"]
        })
    );
    // OpenAI-style arguments, a raw string (ReAct agents), a JSON string, a bare number.
    assert_eq!(tool.call(r#"{"input": "hi"}"#).await.unwrap(), "HI");
    assert_eq!(tool.call("hello there").await.unwrap(), "HELLO THERE");
    assert_eq!(tool.call(r#""quoted""#).await.unwrap(), "QUOTED");
    assert_eq!(tool.call("42").await.unwrap(), "42");
}

#[tokio::test]
async fn function_tool_can_capture_state_and_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>(_: &T) {}
    let log = Arc::new(AsyncMutex::new(Vec::<String>::new()));
    let sink = log.clone();
    let tool = FunctionTool::text("note", "Saves a note", move |s: String| {
        let sink = sink.clone();
        async move {
            sink.lock().await.push(s);
            Ok::<_, std::io::Error>("saved")
        }
    });
    assert_send_sync(&tool);
    let tool: Arc<dyn Tool> = Arc::new(tool);
    let handles: Vec<_> = (0..3)
        .map(|i| {
            let tool = tool.clone();
            tokio::spawn(async move { tool.call(&format!("note {i}")).await.unwrap() })
        })
        .collect();
    for h in handles {
        assert_eq!(h.await.unwrap(), "saved");
    }
    let mut notes = log.lock().await.clone();
    notes.sort();
    assert_eq!(notes, ["note 0", "note 1", "note 2"]);
}

/// End to end through the tools agent: the schema is advertised to the model and the
/// model's arguments reach the function typed.
#[tokio::test]
async fn works_in_the_openai_tools_agent() {
    let calls = json!([{
        "id": "call_1", "type": "function",
        "function": {"name": "adder", "arguments": r#"{"a": 40, "b": 2}"#}
    }])
    .to_string();
    let llm = FakeLLM::new([calls, "It's 42".to_string()]);
    let agent = OpenAiToolAgentBuilder::new()
        .tools(&[Arc::new(adder())])
        .build(llm.clone())
        .unwrap();
    let out = AgentExecutor::from_agent(agent)
        .with_memory(Arc::new(AsyncMutex::new(SimpleMemory::new())))
        .invoke(prompt_args! {"input" => "what's 40 + 2?"})
        .await
        .unwrap();
    assert_eq!(out, "It's 42");

    let functions = llm.options.lock().unwrap()[0].functions.clone().unwrap();
    assert_eq!(functions.len(), 1);
    assert_eq!(functions[0].name, "adder");
    assert_eq!(functions[0].description, "Adds two numbers together");
    assert_eq!(functions[0].parameters, add_schema());

    let observation = llm.calls()[1]
        .iter()
        .find(|m| m.message_type == MessageType::ToolMessage)
        .map(|m| m.content.clone());
    assert_eq!(observation.as_deref(), Some("The sum of 40 and 2 is 42"));
}
