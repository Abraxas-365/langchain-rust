# 🦜️🔗LangChain Rust

[![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/langchain-rust.svg
[crates.io]: https://crates.io/crates/langchain-rust

⚡ Building applications with LLMs through composability, with Rust! ⚡

[![Discord](https://dcbadge.vercel.app/api/server/JJFcTFbanu?style=for-the-badge)](https://discord.gg/JJFcTFbanu)
[![Docs: docs.rs](https://img.shields.io/docsrs/langchain-rust?style=for-the-badge)](https://docs.rs/langchain-rust)
[![Examples](https://img.shields.io/badge/examples-github-success?style=for-the-badge)](https://github.com/Abraxas-365/langchain-rust/tree/main/examples)

## 🤔 What is this?

This is the Rust language implementation of [LangChain](https://github.com/langchain-ai/langchain).

## Current Features

- LLMs

  - [x] [OpenAi](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_openai.rs)
  - [x] [Azure OpenAi](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_azure_open_ai.rs)
  - [x] [Ollama](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_ollama.rs)
  - [x] [Anthropic Claude](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_anthropic_claude.rs)
  - [x] [Response caching for any LLM](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_cache.rs)
  - [x] [Reasoning / thinking (Claude, OpenAI, Ollama, DeepSeek)](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_thinking.rs)
  - [x] [Prompt caching (Claude; cache usage for OpenAI, DeepSeek, Qwen, Bedrock)](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_prompt_caching.rs)

- Embeddings

  - [x] [OpenAi](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/embedding_openai.rs)
  - [x] [Azure OpenAi](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/embedding_azure_open_ai.rs)
  - [x] [Ollama](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/embedding_ollama.rs)
  - [x] [Local FastEmbed](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/embedding_fastembed.rs)
  - [x] [Voyage AI](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/embedding_voyage.rs)
  - [x] [MistralAI](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/embedding_mistralai.rs)
  - [x] [AWS Bedrock (Titan, Cohere)](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/embedding_bedrock.rs)

- VectorStores

  - [x] [MongoDB Atlas](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_mongodb.rs)
  - [x] [OpenSearch](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_opensearch.rs)
  - [x] [Pinecone](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_pinecone.rs)
  - [x] [Postgres](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_postgres.rs)
  - [x] [Qdrant](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_qdrant.rs)
  - [x] [Redis](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_redis.rs)
  - [x] [Chroma](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_chroma.rs)
  - [x] [Sqlite](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_sqlite_vss.rs)
  - [x] [SurrealDB](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_surrealdb/src/main.rs)
  - [x] [Weaviate](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_weaviate.rs)
  - [x] [Milvus](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_milvus.rs)
  - [x] [MariaDB](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/vector_store_mariadb.rs)

- Chain

  - [x] [LLM Chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_chain.rs)
  - [x] [Conversational Chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/conversational_chain.rs)
  - [x] [Conversational Retriever Simple](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/conversational_retriever_simple_chain.rs)
  - [x] [Conversational Retriever With Vector Store](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/conversational_retriever_chain_with_vector_store.rs)
  - [x] [Sequential Chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/sequential_chain.rs)
  - [x] [Q&A Chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/qa_chain.rs)
  - [x] [Map-reduce / refine / map-rerank + Retrieval QA](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/summarize_documents.rs)
  - [x] [SQL Chain](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/sql_chain.rs)
  - [x] [LLM Math Chain (word problems, exact arithmetic)](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/llm_math.rs)
  - [x] [Router Chain (LLM picks the chain)](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/router_chain.rs)
  - [x] [Constitutional Chain (critique and revise against principles)](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/constitutional_chain.rs)
  - [x] [Transform Chain (your own function as a chain step)](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/transform_chain.rs)
  - [x] [API Chain (answer from an HTTP API, allowed URLs only)](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/api_chain.rs)

- Output parsers

  - [x] [Typed JSON, structured fields, lists, yes/no and regex](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/output_parsers.rs)

- Memory

  - [x] [Persistent memory (SQLite/Postgres) + token buffer](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/persistent_memory.rs)
  - [x] [Redis chat history](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/redis_memory.rs)
  - [x] [MongoDB chat history](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/mongodb_memory.rs)

- Agents

  - [x] [Chat Agent with Tools](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/agent.rs)
  - [x] [Open AI Compatible Tools Agent](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/open_ai_tools_agent.rs)
  - [x] [ReAct (MRKL) agent for models without tool calling, with parse-error retries and intermediate steps](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/react_agent.rs)
  - [x] [Streaming agent events + cancellation](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/streaming_agent.rs)
  - [x] [Tracing spans for LLM calls, tools and agent steps](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/tracing.rs)

- Tools

  - [x] [Custom tools from async functions (`FunctionTool`)](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/function_tool.rs)
  - [x] [Calculator](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/calculator_tool.rs)
  - [x] Serpapi/Google
  - [x] DuckDuckGo Search
  - [x] [Exa web search](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/exa_search.rs)
  - [x] [Wolfram/Math](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/wolfram_tool.rs)
  - [x] Command line
  - [x] [Text2Speech](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/speech2text_openai.rs)
  - [x] Wikipedia
  - [x] [Perplexity web search](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/perplexity_search.rs)

- Semantic Routing

  - [x] [Static Routing](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/semantic_routes.rs)
  - [x] [Dynamic Routing](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/dynamic_semantic_routes.rs)

- Document Loaders

  - [x] PDF

    ```rust
    use futures_util::StreamExt;

    async fn main() {
        let path = "./src/document_loaders/test_data/sample.pdf";

        let loader = PdfExtractLoader::from_path(path).expect("Failed to create PdfExtractLoader");
        // let loader = LoPdfLoader::from_path(path).expect("Failed to create LoPdfLoader");

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

  - [x] HTML To Markdown

    ```rust
    use futures_util::StreamExt;
    use url::Url;

    async fn main() {
        let path = "./src/document_loaders/test_data/example.html";
        let html_to_markdown_loader = HtmlToMarkdownLoader::from_path(path, Url::parse("https://example.com/").unwrap(), HtmlToMarkdownOptions::default().with_skip_tags(vec!["figure".to_string()]))
            .expect("Failed to create html to markdown loader");

        let documents = html_to_markdown_loader
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

  - [x] Git commits

    ```rust
    use futures_util::StreamExt;

    async fn main() {
        let path = "/path/to/git/repo";
        let git_commit_loader = GitCommitLoader::from_path(path).expect("Failed to create git commit loader");

        let documents = csv_loader
            .load()
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await;
    }
    ```

  - [x] Source code

    ```rust

    let loader_with_dir =
    SourceCodeLoader::from_path("./src/document_loaders/test_data".to_string())
    .with_dir_loader_options(DirLoaderOptions {
    glob: None,
    suffixes: Some(vec!["rs".to_string()]),
    exclude: None,
    });

    let stream = loader_with_dir.load().await.unwrap();
    let documents = stream.map(|x| x.unwrap()).collect::<Vec<_>>().await;
    ```

  - [x] [AssemblyAI audio transcripts](https://github.com/Abraxas-365/langchain-rust/blob/main/examples/assemblyai_loader.rs)
  - [x] Notion export (`NotionDirectoryLoader::new("./Export-...")`, Markdown & CSV export)

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

##### sqlite-vss

Download additional sqlite_vss libraries from <https://github.com/asg017/sqlite-vss>

```bash
cargo add langchain-rust --features sqlite-vss
```

##### sqlite-vec

Download additional sqlite_vec libraries from <https://github.com/asg017/sqlite-vec>

```bash
cargo add langchain-rust --features sqlite-vec
```


#### With Postgres

```bash
cargo add langchain-rust --features postgres
```

#### With SurrialDB

```bash
cargo add langchain-rust --features surrealdb
```

#### With Weaviate

```bash
cargo add langchain-rust --features weaviate
```

#### With Milvus

```bash
cargo add langchain-rust --features milvus
```

#### With Pinecone

```bash
cargo add langchain-rust --features pinecone
```

#### With Qdrant

```bash
cargo add langchain-rust --features qdrant
```

#### With Chroma

```bash
cargo add langchain-rust --features chroma
```

#### With MongoDB (Atlas vector store, chat history on any MongoDB)

```bash
cargo add langchain-rust --features mongodb
```

#### With Redis (chat history and vector store)

```bash
cargo add langchain-rust --features redis
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
    // let open_ai = OpenAI::default()
    //     .with_config(
    //         OpenAIConfig::default()
    //             .with_api_key("<your_key>"),
    //     ).with_model(OpenAIModel::Gpt4oMini.to_string());
    let open_ai = OpenAI::default().with_model(OpenAIModel::Gpt4oMini.to_string());


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
