use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AgentAction {
    pub id: String,
    pub action: String,
    pub action_input: Value,
}

#[derive(Debug)]
pub enum AgentEvent {
    Action(Vec<AgentAction>),
    Finish(String),
}

impl<'de> Deserialize<'de> for AgentEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut value = Value::deserialize(deserializer)?;

        if let (Some(Value::String(action)), Some(action_input)) = (
            value.get_mut("action").map(|v| v.take()),
            value.get_mut("action_input").map(|v| v.take()),
        ) {
            Ok(AgentEvent::Action(vec![AgentAction {
                id: value
                    .get_mut("id")
                    .and_then(|v| Some(v.take().as_str()?.to_string()))
                    .unwrap_or(uuid::Uuid::new_v4().to_string()),
                action,
                action_input,
            }]))
        } else if let Some(final_answer) = value.get_mut("final_answer").map(|v| v.take()) {
            match final_answer {
                Value::String(value) => return Ok(AgentEvent::Finish(value)),
                v => Ok(AgentEvent::Finish(v.to_string())),
            }
        } else {
            Err(serde::de::Error::custom("Invalid format")) // TODO: provide clearer error message
        }
    }
}

pub enum AgentPlan {
    Text(AgentEvent),
    Stream(mpsc::Receiver<Result<String, reqwest_eventsource::Error>>),
}
