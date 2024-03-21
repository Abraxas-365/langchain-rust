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
    pub log: String, //esto es el proceso de la ia antes de la respuesta del tool Osea 'debo usar esra herramiensta para saber xxx {tool:xxx,input:yyy}'
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LogTools {
    pub tool_id: String,
    pub tools: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AgentFinish {
    pub output: String,
}

pub enum AgentEvent {
    Action(AgentAction),
    Finish(AgentFinish),
}

pub enum AgentPlan {
    Text(AgentEvent),
    Stream(mpsc::Receiver<Result<String, reqwest_eventsource::Error>>),
}
