# 🦜️🔗LangChain Rust

[![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/langchain-rust.svg
[crates.io]: https://crates.io/crates/langchain-rust

⚡ Building applications with LLMs through composability, with Rust! ⚡

[![Discord](https://dcbadge.vercel.app/api/server/JJFcTFbanu?style=for-the-badge)](https://discord.gg/JJFcTFbanu)
[![Docs: Tutorial](https://img.shields.io/badge/docs-tutorial-success?style=for-the-badge&logo=appveyor)](https://langchain-rust.sellie.tech/get-started/quickstart)

## 🤔 What is this?

This is the Rust language implementation of [LangChain](https://github.com/langchain-ai/langchain).

## Current Features

- LLMs

  - [x] [OpenAi](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_openai.rs)
  - [x] [Azure OpenAi](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_azure_open_ai.rs)
  - [x] [Ollama and Compatible Api](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_ollama.rs)
  - [x] [Anthropic Claude](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_anthropic_claude.rs)

- Embeddings

  - [x] [OpenAi](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/embedding_openai.rs)
  - [x] [Ollama](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/embedding_ollama.rs)
  - [x] [Azure OpenAi](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/embedding_azure_open_ai.rs)
  - [x] [Local FastEmbed](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/embedding_fastembed.rs)

- VectorStores

  - [x] [Postgres](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_postgres.rs)
  - [x] [Sqlite](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_sqlite.rs)
  - [x] [SurrealDB](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_surrealdb/src/main.rs)

- Chain

  - [x] [LLM Chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_chain.rs)
  - [x] [Sequential Chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/sequential_chain.rs)
  - [x] [Conversational Chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/conversational_chain.rs)
  - [x] [Q&A Chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/qa_chain.rs)
  - [x] [SQL Chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/sql_chain.rs)

- Agents

  - [x] [Chat Agent with Tools](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/agent.rs)
  - [x] [Open AI Compatible Tools Agent](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/open_ai_tools_agent.rs)

- Tools

  - [x] Serpapi/Google
  - [x] [Wolfram/Math](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/wolfram_tool.rs)
  - [x] Command line
  - [x] [Text2Speech](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/speec2text_openai.rs)

- Semantic Routing

  - [x] [Static Routing](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/semantic_routes.rs)
  - [x] [Dynamic Routing](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/dynamic_semantic_routes.rs)

- Document Loaders

  - [x] PDF

    ```rust
    use futures_util::StreamExt;

    async fn main() {
        let path = "./src/document_loaders/test_data/sample.pdf";

        let loader = LoPdfLoader::from_path(path).expect("Failed to create PdfLoader");

        let docs = loader
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;

    }
    ```

  - [x] Pandoc

    ```rust
    use futures_util::StreamExt;

    async fn main() {

        let path = "./src/document_loaders/test_data/sample.docx";

        let loader = PandocLoader::from_path(InputFormat::Docx.to_string(), path)
            .await
            .expect("Failed to create PandocLoader");

        let docs = loader
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;
    }
    ```

  - [x] HTML

    ```rust
    use futures_util::StreamExt;
    use url::Url;

    async fn main() {
        let path = "./src/document_loaders/test_data/example.html";
        let html_loader = HtmlLoader::from_path(path, Url::parse("https://example.com/").unwrap())
            .expect("Failed to create html loader");

        let documents = html_loader
            .load()
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await;
    }
    ```

  - [x] CSV

    ```rust
    use futures_util::StreamExt;

    async fn main() {
        let path = "./src/document_loaders/test_data/test.csv";
        let columns = vec![
            "name".to_string(),
            "age".to_string(),
            "city".to_string(),
            "country".to_string(),
        ];
        let csv_loader = CsvLoader::from_path(path, columns).expect("Failed to create csv loader");

        let documents = csv_loader
            .load()
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await;
    }
    ```

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

#### With Sqlite

```bash
cargo add langchain-rust --features sqlite
```

Download additional sqlite_vss libraries from https://github.com/asg017/sqlite-vss

#### With Postgres

```bash
cargo add langchain-rust --features postgres
```

#### With SurrialDB

```bash
cargo add langchain-rust --features surrealdb
```

Please remember to replace the feature flags `sqlite`, `postgres` or `surrealdb` based on your
specific use case.

This will add both `serde_json` and `langchain-rust` as dependencies in your `Cargo.toml`
file. Now, when you build your project, both dependencies will be fetched and compiled, and will be available for use in your project.

Remember, `serde_json` is a necessary dependencies, and `sqlite`, `postgres` and `surrealdb`
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
