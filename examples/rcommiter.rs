use std::io::{self, BufRead};
use std::process::{Command, Stdio};

use langchain_rust::chain::chain_trait::Chain;
use langchain_rust::chain::llm_chain::LLMChainBuilder;
use langchain_rust::llm::openai::{OpenAI, OpenAIConfig};
use langchain_rust::prompt::HumanMessagePromptTemplate;
use langchain_rust::{prompt_args, template_jinja2};

//to try this in action , add something to this file stage it an run it
#[tokio::main]
async fn main() -> io::Result<()> {
    let llm = OpenAI::new(OpenAIConfig::default());
    let chain = LLMChainBuilder::new()
        .prompt(HumanMessagePromptTemplate::new(template_jinja2!(
            r#"
    Create a conventional commit message for the following changes.

    File changes: 
        {{input}}



    "#,
            "input"
        )))
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
        .invoke(prompt_args! {
            "input"=>complete_changes,
        })
        .await
        .expect("Failed to invoke chain");

    println!("{}", res);
    Ok(())
}
