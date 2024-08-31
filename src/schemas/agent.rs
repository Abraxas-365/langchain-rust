use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

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
