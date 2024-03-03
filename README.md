# 🦜️🔗LangChain Rust

[![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/langchain-rust.svg
[crates.io]: https://crates.io/crates/langchain-rust

⚡ Building applications with LLMs through composability, with Rust! ⚡

[![Discord](https://dcbadge.vercel.app/api/server/fe3PPm69?style=for-the-badge)](https://discord.gg/JJFcTFbanu)
[![Docs: Tutorial](https://img.shields.io/badge/docs-tutorial-success?style=for-the-badge&logo=appveyor)](https://langchain-rust.sellie.tech/get-started/quickstart)

## 🤔 What is this?

This is the Rust language implementation of [LangChain](https://github.com/langchain-ai/langchain).

# Version

_PLS use the version 2.x.x_ This is the most stable release

# Examples

[rcommit](https://github.com/Abraxas-365/rcommit): rcommit allows you to create git commits with AI

# Quickstart

## Setup

### Installation

```bash
cargo add langchain-rust
```

### Building with LangChain[​](https://python.langchain.com/docs/get_started/quickstart#building-with-langchain) <a href="#building-with-langchain" id="building-with-langchain"></a>

LangChain enables building application that connect external sources of data and computation to LLMs. In this quickstart, we will walk through a few different ways of doing that. We will start with a simple LLM chain, which just relies on information in the prompt template to respond. Finally, we will build an agent - which utilizes an LLM to determine whether or not it needs to use a tool like a calculator.

### LLM Chain[​](https://python.langchain.com/docs/get_started/quickstart#llm-chain) <a href="#llm-chain" id="llm-chain"></a>

We'll show how to use models available via API, like OpenAI.

Accessing the API requires an API key, which you can get by creating an account and heading [here](https://platform.openai.com/account/api-keys). Once we have a key we'll want to set it as an environment variable by running:

```bash
export OPENAI_API_KEY="..."
```

We can then initialize the model:

```rust
let open_ai = OpenAI::default();
```

If you'd prefer not to set an environment variable you can pass the key in directly via the `openai_api_key` named parameter when initiating the OpenAI LLM class:

```rust
let open_ai = OpenAI::default().with_api_key("...");
```

Once you've installed and initialized the LLM of your choice, we can try using it! Let's ask it what LangSmith is - this is something that wasn't present in the training data so it shouldn't have a very good response.

```rust
let resp=open_ai.invoke("how can langsmith help with testing?").await.unwrap();
```

We can also guide it's response with a prompt template. Prompt templates are used to convert raw user input to a better input to the LLM.

```rust
let prompt = message_formatter![
        fmt_message!(Message::new_system_message(
            "You are world class technical documentation writer."
        )),
        fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
            "{input}", "input"
        )))
 ];
```

We can now combine these into a simple LLM chain:

```rust
 let chain = LLMChainBuilder::new()
            .prompt(formatter)
            .llm(llm)
            .build();

```

We can now invoke it and ask the same question. It still won't know the answer, but it should respond in a more proper tone for a technical writer!

```rust
match chain.invoke(prompt_args! {
 "input" => "Quien es el escritor de 20000 millas de viaje submarino",
    }).await {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
```

If you want to prompt to have a list of messages you could use the `fmt_placeholder` macro

```rust
let prompt = message_formatter![
        fmt_message!(Message::new_system_message(
            "You are world class technical documentation writer."
        )),
        fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
            "{input}", "input"
        ))),
        fmt_placeholder!("history"),
 ];
```

And when calling to the chain send the message

```rust
match chain.invoke(prompt_args! {
    "input" => "Quien es el escritor de 20000 millas de viaje submarino",
    "history" => vec![
            Message::new_human_message("Mi nombre es: luis"),
            Message::new_ai_message("Mucho gusto luis"),
            ],

    }).await {
            Ok(result) => {
                println!("Result: {:?}", result);
            }
            Err(e) => panic!("Error invoking LLMChain: {:?}", e),
        }
```

### Conversational Chain

Now we well create a conversational chain with memory, by default , the conversation chain comes with a simple memory, will be inject that as an example, if you dont want the conversation chain to have memory you could inject the `DummyMemroy`

```rust
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
    let memory=SimpleMemory::new();
    let chain = ConversationalChainBuilder::new()
        .llm(llm)
        .memory(memory.into())
        .build()
        .expect("Error building ConversationalChain");

    let input_variables = prompt_args! {
        "input" => "Soy de peru",
    };
    match chain.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }

    let input_variables = prompt_args! {
        "input" => "Cuales son platos tipicos de mi pais",
    };
    match chain.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
}
```

### Sequential Chain [Still Experimental]

The sequential chain allows you to pass the result of n chain to the n+1 chain as input.

```rust
let llm = OpenAI::default();
let chain1 = LLMChainBuilder::new()
    .prompt(template_fstring!(
        "dame un nombre para una tienda de {input}",
        "input"
    ))
    .llm(llm.clone())
    .build()
    .expect("Failed to build LLMChain");

let chain2 = LLMChainBuilder::new()
    .prompt(template_fstring!(
        "dame un slogan para una tienda llamada {output}",
        "output"
    ))
    .llm(llm.clone())
    .build()
    .expect("Failed to build LLMChain");

let chain = sequential_chain!(chain1, chain2);

println!(
    "{:?}",
    chain.call(prompt_args! {"input"=>"medias"}).await.unwrap()
);
```

### Agent

We've so far create examples of chains - where each step is known ahead of time. The final thing we will create is an agent - where the LLM decides what steps to take.

First whe sould create a tool, i will create a mock tool

```rust
struct Calc {}

#[async_trait]
impl Tool for Calc {
    fn name(&self) -> String {
        "Calculator".to_string()
    }
    fn description(&self) -> String {
        "Usefull to make calculations".to_string()
    }
    async fn call(&self, _input: &str) -> Result<String, Box<dyn Error>> {
        Ok("25".to_string())
    }
}
```

\
Then whe create the agent with memory

```rust
  let llm = OpenAI::default().with_model(OpenAIModel::Gpt4);
    let memory = SimpleMemory::new();
    let tool_calc = Calc {};
    let agent = ConversationalAgentBuilder::new()
        .tools(vec![Arc::new(tool_calc)])
        .output_parser(ChatOutputParser::new().into())
        .build(llm)
        .unwrap();
    let input_variables = prompt_args! {
        "input" => "hola,Me llamo luis, y tengo 10 anos, y estudio Computer scinence",
    };
    let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());
    match executor.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
    let input_variables = prompt_args! {
        "input" => "cuanta es la edad de luis +10 y que estudia",
    };
    match executor.invoke(input_variables).await {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    }
```

### Vectore Stores

The vector stores allow you to save embedding data to a database, and retrive it
Using embedding you can search base on `context`

```rust
async fn db_test() {
    let embedder = OpenAiEmbedder::default();
    let store = StoreBuilder::new()
        .embedder(embedder)
        .connection_url("postgresql://postgres:postgres@localhost:5432/postgres")
        .vector_dimensions(1536)
        .build()
        .await
        .unwrap();
    let document = Document {
        page_content: "this is a test".to_string(),
        metadata: HashMap::new(),
        score: 0.0,
    };
    let docs = vec![document];

    store
        .add_documents(&docs, &VecStoreOptions::default())
        .await
        .unwrap();
    println!("Result:");

    let similar = store
        .similarity_search("this is a test", 10, &VecStoreOptions::default())
        .await
        .unwrap();
    println!("Similar: {:?}", similar);
}
```

Or you can use the powerful macros

```rust
async fn db_test() {
    let embedder = OpenAiEmbedder::default();
    let store = StoreBuilder::new()
        .embedder(embedder)
        .connection_url("postgresql://postgres:postgres@localhost:5432/postgres")
        .vector_dimensions(1536)
        .build()
        .await
        .unwrap();
    let document = Document {
        page_content: "this is a test".to_string(),
        metadata: HashMap::new(),
        score: 0.0,
    };
    let docs = vec![document];

    //If you want to send options:
    //add_documents!(store, &docs,&VecStoreOptions::default()).await.unwrap();
    add_documents!(store, &docs).await.unwrap();

    //If you want to send options:
    //similarity_search!(store, "this is a test", 10,&VecStoreOptions::default()).await.unwrap();
    let similar = similarity_search!(store, "this is a test", 10)
        .await
        .unwrap();
    println!("Similar: {:?}", similar);
}
```
