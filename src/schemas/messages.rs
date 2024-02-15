use serde_json::Value;
use std::{
    collections::HashMap,
    io::{self, ErrorKind},
    sync::Arc,
};

use serde::{ser::SerializeStruct, Deserialize, Serialize, Serializer};

pub trait BaseMessage: Send + Sync {
    fn get_type(&self) -> String;
    fn get_content(&self) -> String;
}

pub trait SerializableMessage {
    fn serialize_message<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error>;
}

impl SerializableMessage for Arc<dyn BaseMessage> {
    fn serialize_message<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("BaseMessage", 2)?;
        state.serialize_field("type", &self.get_type())?;
        state.serialize_field("content", &self.get_content())?;
        state.end()
    }
}

//TODO: Implement deserialize

#[derive(Clone, Serialize, Deserialize)]
pub struct HumanMessage {
    pub content: String,
}
impl HumanMessage {
    pub fn new(content: &str) -> Arc<Self> {
        Arc::new(Self {
            content: String::from(content),
        })
    }
}
impl BaseMessage for HumanMessage {
    fn get_type(&self) -> String {
        String::from("user")
    }

    fn get_content(&self) -> String {
        self.content.clone()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    pub content: String,
}
impl SystemMessage {
    pub fn new(content: &str) -> Arc<Self> {
        Arc::new(Self {
            content: String::from(content),
        })
    }
}
impl BaseMessage for SystemMessage {
    fn get_type(&self) -> String {
        String::from("system")
    }

    fn get_content(&self) -> String {
        self.content.clone()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AIMessage {
    pub content: String,
}
impl AIMessage {
    pub fn new(content: &str) -> Arc<Self> {
        Arc::new(Self {
            content: String::from(content),
        })
    }
}
impl BaseMessage for AIMessage {
    fn get_type(&self) -> String {
        String::from("assistant")
    }

    fn get_content(&self) -> String {
        self.content.clone()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    role: String,
    content: String,
}
impl ChatMessage {
    pub fn new(role: &str, content: &str) -> Arc<Self> {
        Arc::new(Self {
            role: String::from(role),
            content: String::from(content),
        })
    }
}
impl BaseMessage for ChatMessage {
    fn get_type(&self) -> String {
        self.role.clone()
    }

    fn get_content(&self) -> String {
        self.content.clone()
    }
}

pub fn message_from_map(
    message: &HashMap<String, String>,
) -> Result<Arc<dyn BaseMessage>, Box<dyn std::error::Error + Send>> {
    let message_type = match message.get("type") {
        Some(t) => t,
        None => {
            return Err(Box::new(io::Error::new(
                ErrorKind::Other,
                "No type key on map",
            )))
        }
    };

    match message_type.as_str() {
        "user" => {
            let content = message.get("content").unwrap_or(&String::from("")).clone();
            Ok(Arc::new(HumanMessage {
                content: content.to_string(),
            }))
        }

        "system" => {
            let content = message.get("content").unwrap_or(&String::from("")).clone();
            Ok(Arc::new(SystemMessage {
                content: content.to_string(),
            }))
        }

        "assistant" => {
            let content = message.get("content").unwrap_or(&String::from("")).clone();
            Ok(Arc::new(AIMessage {
                content: content.to_string(),
            }))
        }

        _ => Err(Box::new(io::Error::new(
            ErrorKind::Other,
            format!("Got unexpected message type: {}", message_type),
        ))),
    }
}

pub fn messages_from_map(
    messages: &[HashMap<String, String>],
) -> Result<Vec<Arc<dyn BaseMessage>>, Box<dyn std::error::Error + Send>> {
    messages.into_iter().map(message_from_map).collect()
}

pub fn message_to_map(message: &Arc<dyn BaseMessage>) -> HashMap<String, String> {
    let mut map = HashMap::new();

    map.insert("type".to_string(), message.get_type());
    map.insert("content".to_string(), message.get_content());

    map
}

pub fn messages_to_map(messages: &[Arc<dyn BaseMessage>]) -> Vec<HashMap<String, String>> {
    messages.into_iter().map(message_to_map).collect()
}

pub fn is_base_message(value: &Value) -> bool {
    if let Some(obj) = value.as_object() {
        let mut message_map = HashMap::new();
        for (k, v) in obj {
            if let Some(string_val) = v.as_str() {
                message_map.insert(k.clone(), string_val.to_string());
            } else {
                return false;
            }
        }

        message_from_map(&message_map).is_ok()
    } else {
        false
    }
}
