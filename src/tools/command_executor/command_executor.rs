use std::error::Error;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::tools::Tool;

pub struct CommandExecutor {
    platform: String,
}

impl CommandExecutor {
    /// Create a new CommandExecutor instance
    /// # Example
    /// ```rust,ignore
    /// let tool = CommandExecutor::new("linux");
    /// ```
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

#[derive(Deserialize, Serialize, Debug)]
struct CommandInput {
    cmd: String,
    #[serde(default)]
    args: Vec<String>,
}
#[derive(Serialize, Deserialize, Debug)]
struct CommandsWrapper {
    commands: Vec<CommandInput>,
}

#[async_trait]
impl Tool for CommandExecutor {
    fn name(&self) -> String {
        String::from("Command_Executor")
    }
    fn description(&self) -> String {
        format!(
            r#""This tool let you run command on the terminal"
            "The input should be an array with comands for the following platform: {}"
            "examle of input: [{{ "cmd": "ls", "args": [] }},{{"cmd":"mkdir","args":["test"]}}]"
            "Should be a comma separeted comands"
            "#,
            self.platform
        )
    }

    fn parameters(&self) -> Value {
        let prompt = format!(
            "This tool let you run command on the terminal.
        The input should be an array with comands for the following platform: {}",
            self.platform
        );
        json!(

        {
          "description": prompt,
          "type": "object",
          "properties": {
            "commands": {
              "description": "An array of command objects to be executed",
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "cmd": {
                    "type": "string",
                    "description": "The command to execute"
                  },
                  "args": {
                    "type": "array",
                    "items": {
                      "type": "string"
                    },
                    "default": [],
                    "description": "List of arguments for the command"
                  }
                },
                "required": ["cmd"],
                "additionalProperties": false,
                "description": "Object representing a command and its optional arguments"
              }
            }
          },
          "required": ["commands"],
          "additionalProperties": false
        }
                )
    }

    async fn parse_input(&self, input: &str) -> Value {
        log::info!("Parsing input: {}", input);

        // Attempt to parse input string into CommandsWrapper struct first
        let wrapper_result = serde_json::from_str::<CommandsWrapper>(input);

        if let Ok(wrapper) = wrapper_result {
            // If successful, serialize the `commands` back into a serde_json::Value
            // this is for llm like open ai tools
            serde_json::to_value(wrapper.commands).unwrap_or_else(|err| {
                log::error!("Serialization error: {}", err);
                Value::Null
            })
        } else {
            // If the first attempt fails, try parsing it as Vec<CommandInput> directly
            // This works on any llm
            let commands_result = serde_json::from_str::<Vec<CommandInput>>(input);

            commands_result.map_or_else(
                |err| {
                    log::error!("Failed to parse input: {}", err);
                    Value::Null
                },
                |commands| serde_json::to_value(commands).unwrap_or(Value::Null),
            )
        }
    }

    async fn run(&self, input: Value) -> Result<String, Box<dyn Error>> {
        let commands: Vec<CommandInput> = serde_json::from_value(input)?;
        let mut result = String::new();

        for command in commands {
            let mut command_to_execute = std::process::Command::new(&command.cmd);
            command_to_execute.args(&command.args);

            let output = command_to_execute.output()?;

            result.push_str(&format!(
                "Command: {}\nOutput: {}",
                command.cmd,
                String::from_utf8_lossy(&output.stdout),
            ));

            if !output.status.success() {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!(
                        "Command {} failed with status: {}",
                        command.cmd, output.status
                    ),
                )));
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use serde_json::json;
    #[tokio::test]
    async fn test_with_string_executor() {
        let tool = CommandExecutor::new("linux");
        let input = json!({
            "commands": [
                {
                    "cmd": "ls",
                    "args": []
                }
            ]
        });
        println!("{}", &input.to_string());
        let result = tool.call(&input.to_string()).await.unwrap();
        println!("Res: {}", result);
    }
}
