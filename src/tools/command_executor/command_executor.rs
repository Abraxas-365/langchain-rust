use std::error::Error;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::tools::Tool;

pub struct CommandExecutor {
    platform: String,
}

impl CommandExecutor {
    pub fn new<S: Into<String>>(platform: S) -> Self {
        Self {
            platform: platform.into(),
        }
    }
}

impl Default for CommandExecutor {
    fn default() -> Self {
        Self::new("linux")
    }
}

#[async_trait]
impl Tool for CommandExecutor {
    fn name(&self) -> String {
        String::from("Command_Executor")
    }
    fn description(&self) -> String {
        String::from(format!(
            r#""This tool let you run command on the terminal"
            "The input should be an array with comands for the following platform: {}"
            "examle of input: [ls, mkdir test]"
            "Should be a comma separeted comands"
            "#,
            self.platform
        ))
    }

    fn parameters(&self) -> Value {
        let prompt = format!(
            "This tool let you run command on the terminal.
        The input should be an array with comands for the following platform: {}",
            self.platform
        );
        json!({
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": prompt,
                }
            },
            "required": ["commands"]
        })
    }

    async fn parse_input(&self, input: &str) -> Value {
        log::info!("Parsing input: {}", input);
        match serde_json::from_str::<Value>(input) {
            Ok(input) => {
                if input["commands"].is_array() {
                    Value::from(
                        input["commands"]
                            .as_array()
                            .unwrap_or(&vec![])
                            .iter()
                            .map(|s| s.as_str().unwrap_or_default().to_string())
                            .collect::<Vec<String>>(),
                    )
                } else {
                    Value::from(
                        input["commands"]
                            .as_str()
                            .unwrap_or_default()
                            .split(",")
                            .map(|s| s.to_string())
                            .collect::<Vec<String>>(),
                    )
                }
            }
            Err(_) => Value::from(
                input
                    .split(",")
                    .map(|s| s.to_string())
                    .collect::<Vec<String>>(),
            ),
        }
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let commands = input
            .as_array()
            .ok_or("Input should be an array")?
            .iter()
            .map(|s| s.as_str().unwrap_or_default())
            .collect::<Vec<&str>>();
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(commands.join(" && "))
            .output()?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_json::json;
    #[tokio::test]
    async fn test_with_value_executor() {
        let tool = CommandExecutor::new("linux");
        let result = tool.call("ls,pwd").await.unwrap();
        println!("{}", result);
    }

    async fn test_with_string_executor() {
        let tool = CommandExecutor::new("linux");
        let input = json!({
            "commands": ["ls", "pwd"]
        });

        let result = tool.call(&input.to_string()).await.unwrap();
        println!("{}", result);
    }
}
