use std::io::{self, BufRead};
use std::process::{Command, Stdio};

use indoc::indoc;
use langchain_rust::chain::chain_trait::Chain;
use langchain_rust::chain::llm_chain::LLMChainBuilder;
use langchain_rust::input_variables;
use langchain_rust::llm::openai::OpenAI;
use langchain_rust::schemas::{MessageTemplate, MessageType};

//to try this in action , add something to this file stage it an run it
#[tokio::main]
async fn main() -> io::Result<()> {
    let prompt = MessageTemplate::from_jinja2(
        MessageType::HumanMessage,
        indoc! {"
            Create a conventional commit message for the following changes.

            File changes: 
                {{input}}


        "},
    );

    let llm = OpenAI::default();
    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(llm)
        .build()
        .expect("Failed to build LLMChain");

    let shell_command = r#"
git diff --cached --name-only --diff-filter=ACM | while read -r file; do echo "\n---------------------------\n name:$file"; git diff --cached "$file" | sed 's/^/changes:/'; done
"#;

    let output = Command::new("sh")
        .arg("-c")
        .arg(shell_command)
        .stdout(Stdio::piped())
        .spawn()?
        .stdout
        .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Could not capture stdout."))?;

    let reader = io::BufReader::new(output);

    let complete_changes = reader
        .lines()
        .map(|line| line.unwrap())
        .collect::<Vec<String>>()
        .join("\n");

    let res = chain
        .invoke(&mut input_variables! {
            "input" => complete_changes,
        })
        .await
        .expect("Failed to invoke chain");

    println!("{}", res);
    Ok(())
}
