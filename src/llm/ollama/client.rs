use crate::{
    language_models::{llm::LLM, options::CallOptions, GenerateResult, LLMError, TokenUsage},
    schemas::{FunctionCallResponse, FunctionDefinition, Message, MessageType, StreamData},
};
use async_trait::async_trait;
use futures::Stream;
use ollama_rs::generation::{
    chat::ChatMessageFinalResponseData,
    images::Image,
    tools::{ToolCall, ToolCallFunction, ToolInfo},
};
pub use ollama_rs::{
    error::OllamaError,
    generation::chat::{request::ChatMessageRequest, ChatMessage, MessageRole},
    models::ModelOptions as GenerationOptions,
    Ollama as OllamaClient,
};
use serde_json::{json, Value};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio_stream::StreamExt;

#[derive(Debug, Clone)]
pub struct Ollama {
    pub(crate) client: Arc<OllamaClient>,
    pub(crate) model: String,
    pub(crate) options: Option<GenerationOptions>,
    /// Tools advertised to the model (set by the tools agent through `add_options`).
    pub(crate) tools: Vec<FunctionDefinition>,
}

/// [llama3.2](https://ollama.com/library/llama3.2) is a 3B parameters, 2.0GB model.
const DEFAULT_MODEL: &str = "llama3.2";

impl Ollama {
    pub fn new<S: Into<String>>(
        client: Arc<OllamaClient>,
        model: S,
        options: Option<GenerationOptions>,
    ) -> Self {
        Ollama {
            client,
            model: model.into(),
            options,
            tools: Vec::new(),
        }
    }

    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_options(mut self, options: GenerationOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Tools the model may call. The tools agent sets these automatically.
    pub fn with_tools(mut self, tools: Vec<FunctionDefinition>) -> Self {
        self.tools = tools;
        self
    }

    fn generate_request(&self, messages: &[Message]) -> Result<ChatMessageRequest, LLMError> {
        let mapped_messages = messages.iter().map(|message| message.into()).collect();
        let mut request = ChatMessageRequest::new(self.model.clone(), mapped_messages);
        if !self.tools.is_empty() {
            request = request.tools(
                self.tools
                    .iter()
                    .map(tool_info)
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        Ok(match &self.options {
            Some(options) => request.options(options.clone()),
            None => request,
        })
    }
}

fn tool_info(f: &FunctionDefinition) -> Result<ToolInfo, LLMError> {
    // ToolInfo holds a schemars Schema; build it from the JSON schema we already have.
    serde_json::from_value(json!({
        "type": "function",
        "function": {
            "name": f.name,
            "description": f.description,
            "parameters": f.parameters,
        }
    }))
    .map_err(|e| LLMError::OtherError(format!("invalid parameters schema for {}: {e}", f.name)))
}

/// Ollama tool calls have no id; the agent needs one to pair results with calls.
fn next_call_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!("call_{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Renders Ollama tool calls as the OpenAI-like JSON array the tools agent reads (same contract
/// as the OpenAI and Claude clients).
fn tool_calls_to_generation(calls: &[ToolCall]) -> String {
    Value::Array(
        calls
            .iter()
            .map(|c| {
                json!({
                    "id": next_call_id(),
                    "type": "function",
                    "function": {
                        "name": c.function.name,
                        "arguments": c.function.arguments.to_string(),
                    },
                })
            })
            .collect(),
    )
    .to_string()
}

fn to_ollama_tool_calls(tool_calls: &Option<Value>) -> Vec<ToolCall> {
    let Some(calls) = tool_calls
        .clone()
        .and_then(|v| serde_json::from_value::<Vec<FunctionCallResponse>>(v).ok())
    else {
        return Vec::new();
    };
    calls
        .into_iter()
        .map(|c| ToolCall {
            function: ToolCallFunction {
                arguments: c.function.arguments_object(),
                name: c.function.name,
            },
        })
        .collect()
}

impl From<&Message> for ChatMessage {
    fn from(message: &Message) -> Self {
        let images = match message.images.clone() {
            Some(images) => {
                let images = images
                    .iter()
                    .map(|image| Image::from_base64(&image.image_url))
                    .collect();
                Some(images)
            }
            None => None,
        };
        let mut chat_message =
            ChatMessage::new(message.message_type.clone().into(), message.content.clone());
        chat_message.images = images;
        if message.message_type == MessageType::AIMessage {
            chat_message.tool_calls = to_ollama_tool_calls(&message.tool_calls);
        }
        chat_message
    }
}

impl From<MessageType> for MessageRole {
    fn from(message_type: MessageType) -> Self {
        match message_type {
            MessageType::AIMessage => MessageRole::Assistant,
            MessageType::ToolMessage => MessageRole::Tool,
            MessageType::SystemMessage => MessageRole::System,
            MessageType::HumanMessage => MessageRole::User,
        }
    }
}

impl Default for Ollama {
    fn default() -> Self {
        let client = Arc::new(OllamaClient::default());
        Ollama::new(client, String::from(DEFAULT_MODEL), None)
    }
}

fn token_usage(final_data: &ChatMessageFinalResponseData) -> TokenUsage {
    let prompt_tokens = final_data.prompt_eval_count as u32;
    let completion_tokens = final_data.eval_count as u32;
    TokenUsage::new(prompt_tokens, completion_tokens)
}

#[async_trait]
impl LLM for Ollama {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let request = self.generate_request(messages)?;
        let result = self.client.send_chat_messages(request).await?;

        let generation = if result.message.tool_calls.is_empty() {
            result.message.content
        } else {
            tool_calls_to_generation(&result.message.tool_calls)
        };

        let tokens = result.final_data.as_ref().map(token_usage);

        Ok(GenerateResult { tokens, generation })
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let request = self.generate_request(messages)?;
        let result = self.client.send_chat_messages_stream(request).await?;

        let stream = result.map(|data| match data {
            Ok(data) => {
                let content = data.message.content.clone();
                let tokens = data.final_data.as_ref().map(token_usage);
                Ok(StreamData::new(
                    serde_json::to_value(data).unwrap_or_default(),
                    tokens,
                    content,
                ))
            }
            Err(_) => Err(OllamaError::Other("Stream error".to_string()).into()),
        });

        Ok(Box::pin(stream))
    }

    /// Only `functions` is used (it becomes the Ollama `tools` list); generation settings go
    /// through [`GenerationOptions`].
    fn add_options(&mut self, options: CallOptions) {
        if let Some(functions) = options.functions {
            self.tools = functions;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;
    use tokio_stream::StreamExt;

    #[tokio::test]
    #[ignore]
    async fn test_generate() {
        let ollama = Ollama::default().with_model("llama3.2");
        let response = ollama.invoke("Hey Macarena, ay").await.unwrap();
        println!("{}", response);
    }

    #[tokio::test]
    #[ignore]
    async fn test_stream() {
        let ollama = Ollama::default().with_model("llama3.2");

        let message = Message::new_human_message("Why does water boil at 100 degrees?");
        let mut stream = ollama.stream(&vec![message]).await.unwrap();
        let mut stdout = tokio::io::stdout();
        while let Some(res) = stream.next().await {
            let data = res.unwrap();
            stdout.write_all(data.content.as_bytes()).await.unwrap();
        }
        stdout.write_all(b"\n").await.unwrap();
        stdout.flush().await.unwrap();
    }

    fn ollama_for(server: &mockito::ServerGuard) -> Ollama {
        let client = Arc::new(OllamaClient::try_new(server.url()).unwrap());
        Ollama::new(client, "m", None)
    }

    #[tokio::test]
    async fn generate_against_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/api/chat")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "model": "m",
                "stream": false,
                "options": {"temperature": 0.5},
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hi"}
                ]
            })))
            .with_header("content-type", "application/json")
            .with_body(
                serde_json::json!({
                    "model": "m",
                    "created_at": "2024-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": "hello"},
                    "done": true,
                    "total_duration": 1,
                    "load_duration": 1,
                    "prompt_eval_count": 3,
                    "prompt_eval_duration": 1,
                    "eval_count": 2,
                    "eval_duration": 1
                })
                .to_string(),
            )
            .create_async()
            .await;

        let result = ollama_for(&server)
            .with_options(GenerationOptions::default().temperature(0.5))
            .generate(&[
                Message::new_system_message("sys"),
                Message::new_human_message("hi"),
            ])
            .await
            .unwrap();
        mock.assert_async().await;
        assert_eq!(result.generation, "hello");
        let tokens = result.tokens.unwrap();
        assert_eq!(
            (
                tokens.prompt_tokens,
                tokens.completion_tokens,
                tokens.total_tokens
            ),
            (3, 2, 5)
        );
    }

    #[tokio::test]
    async fn stream_against_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let line = |content: &str, done: bool| {
            let mut v = serde_json::json!({
                "model": "m",
                "created_at": "2024-01-01T00:00:00Z",
                "message": {"role": "assistant", "content": content},
                "done": done
            });
            if done {
                for (k, n) in [
                    ("total_duration", 1),
                    ("load_duration", 1),
                    ("prompt_eval_count", 3),
                    ("prompt_eval_duration", 1),
                    ("eval_count", 2),
                    ("eval_duration", 1),
                ] {
                    v[k] = n.into();
                }
            }
            v.to_string()
        };
        server
            .mock("POST", "/api/chat")
            .match_body(mockito::Matcher::PartialJson(
                serde_json::json!({"stream": true}),
            ))
            .with_header("content-type", "application/x-ndjson")
            .with_body(format!("{}\n{}\n", line("Hel", false), line("lo", true)))
            .create_async()
            .await;

        let mut stream = ollama_for(&server)
            .stream(&[Message::new_human_message("hi")])
            .await
            .unwrap();
        let mut text = String::new();
        let mut usage = None;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            text.push_str(&chunk.content);
            usage = chunk.tokens.or(usage);
        }
        assert_eq!(text, "Hello");
        assert_eq!(usage.unwrap().total_tokens, 5);
    }

    fn chat_response(message: serde_json::Value) -> String {
        serde_json::json!({
            "model": "m", "created_at": "2024-01-01T00:00:00Z", "message": message,
            "done": true, "total_duration": 1, "load_duration": 1, "prompt_eval_count": 3,
            "prompt_eval_duration": 1, "eval_count": 2, "eval_duration": 1
        })
        .to_string()
    }

    #[test]
    fn tool_history_maps_to_ollama_messages() {
        let calls = serde_json::json!([
            {"id": "call_1", "type": "function",
             "function": {"name": "weather", "arguments": "{\"city\":\"Lima\"}"}},
            {"id": "call_2", "type": "function", "function": {"name": "time", "arguments": "now"}}
        ]);
        let request = Ollama::default()
            .generate_request(&[
                Message::new_ai_message("").with_tool_calls(calls),
                Message::new_tool_message("sunny", "call_1"),
            ])
            .unwrap();
        let v = serde_json::to_value(&request).unwrap();
        assert_eq!(
            v["messages"][0]["tool_calls"],
            serde_json::json!([
                {"function": {"name": "weather", "arguments": {"city": "Lima"}}},
                {"function": {"name": "time", "arguments": {"input": "now"}}}
            ])
        );
        // Tool results use the "tool" role (they used to be sent as "assistant").
        assert_eq!(v["messages"][1]["role"], "tool");
        assert_eq!(v["messages"][1]["content"], "sunny");
        assert!(v.get("tools").is_none());
    }

    #[test]
    fn add_options_sets_tools() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        });
        let mut ollama = Ollama::default();
        ollama.add_options(
            CallOptions::new().with_functions(vec![FunctionDefinition::new(
                "weather",
                "Get the weather",
                schema.clone(),
            )]),
        );
        let v = serde_json::to_value(ollama.generate_request(&[]).unwrap()).unwrap();
        assert_eq!(
            v["tools"],
            serde_json::json!([{"type": "function", "function": {
                "name": "weather", "description": "Get the weather", "parameters": schema
            }}])
        );
    }

    #[tokio::test]
    async fn generate_returns_tool_calls() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/api/chat")
            .with_header("content-type", "application/json")
            .with_body(chat_response(serde_json::json!({
                "role": "assistant", "content": "",
                "tool_calls": [{"function": {"name": "weather", "arguments": {"city": "Lima"}}}]
            })))
            .create_async()
            .await;
        let out = ollama_for(&server).invoke("weather?").await.unwrap();
        let calls: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(calls[0]["type"], "function");
        assert_eq!(calls[0]["function"]["name"], "weather");
        assert_eq!(calls[0]["function"]["arguments"], "{\"city\":\"Lima\"}");
        assert!(calls[0]["id"].as_str().unwrap().starts_with("call_"));
    }

    #[derive(Clone)]
    struct Weather(Arc<std::sync::Mutex<Vec<serde_json::Value>>>);

    #[async_trait]
    impl crate::tools::Tool for Weather {
        fn name(&self) -> String {
            "weather".into()
        }
        fn description(&self) -> String {
            "Current weather for a city".into()
        }
        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            })
        }
        async fn parse_input(&self, input: &str) -> serde_json::Value {
            serde_json::from_str(input).unwrap_or_default()
        }
        async fn run(
            &self,
            input: serde_json::Value,
        ) -> Result<String, Box<dyn std::error::Error>> {
            self.0.lock().unwrap().push(input);
            Ok("sunny".into())
        }
    }

    #[tokio::test]
    async fn tools_agent_runs_a_tool_loop_with_ollama() {
        use crate::{
            agent::{AgentExecutor, OpenAiToolAgentBuilder},
            chain::Chain,
            prompt_args,
            tools::Tool,
        };
        let mut server = mockito::Server::new_async().await;
        // 2nd turn: assistant tool call + tool result are sent back.
        let second = server
            .mock("POST", "/api/chat")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "messages": [
                    {"role": "system"},
                    {"role": "user", "content": "weather in Lima?"},
                    {"role": "assistant", "content": "",
                     "tool_calls": [{"function": {"name": "weather", "arguments": {"city": "Lima"}}}]},
                    {"role": "tool", "content": "sunny"}
                ]
            })))
            .with_header("content-type", "application/json")
            .with_body(chat_response(
                serde_json::json!({"role": "assistant", "content": "Sunny in Lima."}),
            ))
            .expect(1)
            .create_async()
            .await;
        // 1st turn: tools advertised, model asks for the tool.
        let first = server
            .mock("POST", "/api/chat")
            .match_body(mockito::Matcher::PartialJson(serde_json::json!({
                "tools": [{"type": "function", "function": {"name": "weather"}}],
                "messages": [{"role": "system"}, {"role": "user", "content": "weather in Lima?"}]
            })))
            .with_header("content-type", "application/json")
            .with_body(chat_response(serde_json::json!({
                "role": "assistant", "content": "",
                "tool_calls": [{"function": {"name": "weather", "arguments": {"city": "Lima"}}}]
            })))
            .expect(1)
            .create_async()
            .await;

        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let tool: Arc<dyn Tool> = Arc::new(Weather(calls.clone()));
        let agent = OpenAiToolAgentBuilder::new()
            .tools(&[tool])
            .build(ollama_for(&server))
            .unwrap();
        let out = AgentExecutor::from_agent(agent)
            .invoke(prompt_args! {"input" => "weather in Lima?"})
            .await
            .unwrap();

        assert_eq!(out, "Sunny in Lima.");
        assert_eq!(
            *calls.lock().unwrap(),
            vec![serde_json::json!({"city": "Lima"})]
        );
        first.assert_async().await;
        second.assert_async().await;
    }
}
