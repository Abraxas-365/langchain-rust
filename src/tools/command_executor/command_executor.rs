use std::{error::Error, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tools::{
    tool_field::{ArrayField, ObjectField, StringField},
    Tool, ToolFunction, ToolWrapper,
};

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
pub struct CommandInput {
    cmd: String,
    #[serde(default)]
    args: Vec<String>,
}

#[derive(Deserialize, Serialize, Debug)]
struct CommandsInput {
    commands: Vec<CommandInput>,
}

#[async_trait]
impl ToolFunction for CommandExecutor {
    type Input = Vec<CommandInput>;
    type Result = String;

    fn name(&self) -> String {
        String::from("Command_Executor")
    }
    fn description(&self) -> String {
        format!(
            r#""This tool let you run command on the terminal"
            "The input should be an array with commands for the following platform: {}"
            "examle of input: [{{ "cmd": "ls", "args": [] }},{{"cmd":"mkdir","args":["test"]}}]"
            "Should be a comma separated commands"
            "#,
            self.platform
        )
    }

    fn parameters(&self) -> ObjectField {
        ObjectField::new(
            "input",
            None,
            true,
            vec![ArrayField::new(
                "commands",
                Some("An array of command objects to be executed".into()),
                true,
                ObjectField::new(
                    "items",
                    Some("Object representing a command and its optional arguments".into()),
                    true,
                    vec![
                        StringField::new("cmd", Some("The command to execute".into()), true, None)
                            .into(),
                        ArrayField::new_string_array(
                            "args",
                            Some("List of arguments for the command".into()),
                            false,
                        )
                        .into(),
                    ],
                    Some(false),
                )
                .into(),
            )
            .into()],
            Some(false),
        )
    }

    async fn parse_input(
        &self,
        input: Value,
    ) -> Result<Vec<CommandInput>, Box<dyn Error + Send + Sync>> {
        serde_json::from_value::<CommandsInput>(input.clone())
            .map(|commands| commands.commands)
            .or_else(|_| serde_json::from_value::<Vec<CommandInput>>(input))
            .map_err(|e| e.into())
    }

    async fn run(&self, input: Vec<CommandInput>) -> Result<String, Box<dyn Error + Send + Sync>> {
        let commands = input;
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

impl From<CommandExecutor> for Arc<dyn Tool> {
    fn from(val: CommandExecutor) -> Self {
        Arc::new(ToolWrapper::new(val))
    }
}

#[cfg(test)]
mod test {
    use crate::tools::Tool;

    use super::*;
    use serde_json::json;
    #[tokio::test]
    async fn test_with_string_executor() {
        let tool: Arc<dyn Tool> = CommandExecutor::new("linux").into();
        let input = json!({
            "commands": [
                {
                    "cmd": "ls",
                    "args": []
                }
            ]
        });
        println!("{}", &input.to_string());
        let result = tool.call(Value::String(input.to_string())).await.unwrap();
        println!("Res: {}", result);
    }

    #[test]
    fn command_executor_properties_description() {
        let tool = CommandExecutor::new("linux");

        println!("{}", tool.parameters().properties_description());
    }
}
