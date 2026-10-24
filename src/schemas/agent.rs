use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::language_models::ThinkingBlock;

pub enum ToolInput {
    //Will implement this in the future
    StrInput(String),
    DictInput(HashMap<String, String>),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AgentAction {
    pub tool: String,
    pub tool_input: String, //this should be ToolInput in the future
    pub log: String,
}

///Log tools is a struct used by the openai-like agents
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LogTools {
    pub tool_id: String,
    pub tools: String,
    /// The reasoning the model returned with these tool calls. The agent sends it back
    /// with the calls, which Claude requires when thinking is on.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub thinking: Vec<ThinkingBlock>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AgentFinish {
    pub output: String,
}

#[derive(Debug)]
pub enum AgentEvent {
    Action(Vec<AgentAction>),
    Finish(AgentFinish),
}

pub enum AgentPlan {
    Text(AgentEvent),
    Stream(mpsc::Receiver<Result<String, reqwest_eventsource::Error>>),
}
