# 🦜️🔗LangChain Rust

[![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/langchain-rust.svg
[crates.io]: https://crates.io/crates/langchain-rust

⚡ Building applications with LLMs through composability, with Rust! ⚡

[![Discord](https://dcbadge.vercel.app/api/server/JJFcTFbanu?style=for-the-badge)](https://discord.gg/JJFcTFbanu)
[![Docs: Tutorial](https://img.shields.io/badge/docs-tutorial-success?style=for-the-badge&logo=appveyor)](https://langchain-rust.sellie.tech/get-started/quickstart)

## 🤔 What is this?

This is the Rust language implementation of [LangChain](https://github.com/langchain-ai/langchain).

## Examples

- [rcommit](https://github.com/Abraxas-365/rcommit): rcommit allows you to create git commits with AI

- [Conversational chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/conversational_chain.rs) : A chain design for conversation, with memory

- [LLM chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_chain.rs) : the core chain, customisable

- [Secuential chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/sequential_chain.rs): run chain in secuential order, with the ouput of one being the input of the next one

- [Vectore Store with pgvector](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vectore_stores.rs) : And implementation of vector stroe with pgvector

- [Agentes](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/agent.rs) : Agent implementation for complex tasks

- [Open AI Tools Agent](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/open_ai_tools_agent.rs) : Agent using openAi Tools

- [Streaming from a Chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/streaming_from_chain.rs) : Streming example

- [Q&A chian](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/qa_chain.rs) : Question answer Chain

## Installation

This library heavily relies on `serde_json` for its operation.

### Step 1: Add `serde_json`

First, ensure `serde_json` is added to your Rust project.

```bash
cargo add serde_json
```

### Step 2: Add `langchain-rust`

Then, you can add `langchain-rust` to your Rust project.

#### Simple install

```bash
cargo add langchain-rust
```

#### With Postgres

```bash
cargo add langchain-rust --features postgres
```

#### With SurrialDB

```bash
cargo add langchain-rust --features surrealdb
```

Please remember to replace the feature flags `postgres` or `surrealdb` based on your
specific use case.

This will add both `serde_json` and `langchain-rust` as dependencies in your `Cargo.toml`
file. Now, when you build your project, both dependencies will be fetched and compiled, and will be available for use in your project.

Remember, `serde_json` is a necessary dependencies, and `postgres` and `surrealdb`
are optional features that may be added according to project needs.

### Quick Start Conversational Chain

```rust
use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    fmt_message, fmt_placeholder, fmt_template,
    language_models::llm::LLM,
    llm::openai::{OpenAI, OpenAIModel},
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::messages::Message,
    template_fstring,
};

#[tokio::main]
async fn main() {
    //We can then initialize the model:
    // If you'd prefer not to set an environment variable you can pass the key in directly via the `openai_api_key` named parameter when initiating the OpenAI LLM class:
    //let open_ai = OpenAI::default().with_api_key("...");
    let open_ai = OpenAI::default().with_model(OpenAIModel::Gpt35.to_string());

    //Once you've installed and initialized the LLM of your choice, we can try using it! Let's ask it what LangSmith is - this is something that wasn't present in the training data so it shouldn't have a very good response.
    let resp = open_ai.invoke("What is rust").await.unwrap();
    println!("{}", resp);

    // We can also guide it's response with a prompt template. Prompt templates are used to convert raw user input to a better input to the LLM.
    let prompt = message_formatter![
        fmt_message!(Message::new_system_message(
            "You are world class technical documentation writer."
        )),
        fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
            "{input}", "input"
        )))
    ];

    //We can now combine these into a simple LLM chain:

    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(open_ai.clone())
        .build()
        .unwrap();

    //We can now invoke it and ask the same question. It still won't know the answer, but it should respond in a more proper tone for a technical writer!

    match chain
        .invoke(prompt_args! {
        "input" => "Quien es el escritor de 20000 millas de viaje submarino",
           })
        .await
    {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }

    //If you want to prompt to have a list of messages you could use the `fmt_placeholder` macro

    let prompt = message_formatter![
        fmt_message!(Message::new_system_message(
            "You are world class technical documentation writer."
        )),
        fmt_placeholder!("history"),
        fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
            "{input}", "input"
        ))),
    ];

    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(open_ai)
        .build()
        .unwrap();
    match chain
        .invoke(prompt_args! {
        "input" => "Who is the writer of 20,000 Leagues Under the Sea, and what is my name?",
        "history" => vec![
                Message::new_human_message("My name is: luis"),
                Message::new_ai_message("Hi luis"),
                ],

        })
        .await
    {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
```
