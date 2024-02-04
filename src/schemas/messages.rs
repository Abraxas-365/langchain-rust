use serde_json::Value;
use std::{
    collections::HashMap,
    io::{self, ErrorKind},
};

use serde::{Deserialize, Serialize};

pub trait BaseMessage: Send + Sync {
    fn get_type(&self) -> String;
    fn get_content(&self) -> String;
    fn clone_box(&self) -> Box<dyn BaseMessage>;
}
impl Clone for Box<dyn BaseMessage> {
    fn clone(&self) -> Box<dyn BaseMessage> {
        self.clone_box()
    }
}

impl Serialize for Box<dyn BaseMessage> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let map = message_to_map(self);
        map.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Box<dyn BaseMessage> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let map: HashMap<String, String> = HashMap::deserialize(deserializer)?;

        message_from_map(&map).map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct HumanMessage {
    pub content: String,
}
impl HumanMessage {
    pub fn new(content: &str) -> Self {
        Self {
            content: String::from(content),
        }
    }
}
impl BaseMessage for HumanMessage {
    fn get_type(&self) -> String {
        String::from("user")
    }

    fn get_content(&self) -> String {
        self.content.clone()
    }
    fn clone_box(&self) -> Box<dyn BaseMessage> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SystemMessage {
    pub content: String,
}
impl SystemMessage {
    pub fn new(content: &str) -> Self {
        Self {
            content: String::from(content),
        }
    }
}
impl BaseMessage for SystemMessage {
    fn get_type(&self) -> String {
        String::from("system")
    }

    fn get_content(&self) -> String {
        self.content.clone()
    }

    fn clone_box(&self) -> Box<dyn BaseMessage> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct AIMessage {
    pub content: String,
}
impl AIMessage {
    pub fn new(content: &str) -> Self {
        Self {
            content: String::from(content),
        }
    }
}
impl BaseMessage for AIMessage {
    fn get_type(&self) -> String {
        String::from("assistant")
    }

    fn get_content(&self) -> String {
        self.content.clone()
    }

    fn clone_box(&self) -> Box<dyn BaseMessage> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    role: String,
    content: String,
}
impl ChatMessage {
    pub fn new(role: &str, content: &str) -> Self {
        Self {
            role: String::from(role),
            content: String::from(content),
        }
    }
}
impl BaseMessage for ChatMessage {
    fn get_type(&self) -> String {
        self.role.clone()
    }

    fn get_content(&self) -> String {
        self.content.clone()
    }

    fn clone_box(&self) -> Box<dyn BaseMessage> {
        Box::new(self.clone())
    }
}

pub fn message_from_map(
    message: &HashMap<String, String>,
) -> Result<Box<dyn BaseMessage>, Box<dyn std::error::Error + Send>> {
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
            Ok(Box::new(HumanMessage {
                content: content.to_string(),
            }))
        }

        "system" => {
            let content = message.get("content").unwrap_or(&String::from("")).clone();
            Ok(Box::new(SystemMessage {
                content: content.to_string(),
            }))
        }

        "assistant" => {
            let content = message.get("content").unwrap_or(&String::from("")).clone();
            Ok(Box::new(AIMessage {
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
) -> Result<Vec<Box<dyn BaseMessage>>, Box<dyn std::error::Error + Send>> {
    messages.into_iter().map(message_from_map).collect()
}

pub fn message_to_map(message: &Box<dyn BaseMessage>) -> HashMap<String, String> {
    let mut map = HashMap::new();

    map.insert("type".to_string(), message.get_type());
    map.insert("content".to_string(), message.get_content());

    map
}

pub fn messages_to_map(messages: &[Box<dyn BaseMessage>]) -> Vec<HashMap<String, String>> {
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
