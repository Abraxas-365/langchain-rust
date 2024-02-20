# 🦜️🔗LangChain Rust

⚡ Building applications with LLMs through composability, with Rust! ⚡

## 🤔 What is this?

This is the Rust language implementation of [LangChain](https://github.com/langchain-ai/langchain).

## Examples

### LLM

The LLM (Language Model) interface in LangChain Rust provides a standardized way to interact with various language models, facilitating tasks such as text generation and information retrieval. Below is an overview of how to utilize the LLM traits and implement them for specific models like OpenAI's GPT

- Default implementation with open ai

```rust
let options = CallOptions::default();
let open_ai = OpenAI::default();
let messages = vec![Message::new_human_message("Hello, how are you?")];
let resp=open_ai.generate(&messages).await.unwrap();
println!("Generate Result: {:?}", resp);
```

- Implementation with options and streams

```rust
let message_complete = Arc::new(Mutex::new(String::new()));

let streaming_func = {
    let message_complete = message_complete.clone();
    move |content: String| {
        let message_complete = message_complete.clone();
        async move {
            let mut message_complete_lock = message_complete.lock().await;
            println!("Content: {:?}", content);
            message_complete_lock.push_str(&content);
            Ok(())
        }
    }
};
let options = CallOptions::new().with_streaming_func(streaming_func);
let open_ai = OpenAI::new(options).with_model(OpenAIModel::Gpt35); // You can change the model as needed
let messages = vec![Message::new_human_message("Hello, how are you?")];
match open_ai.generate(&messages).await {
    Ok(result) => {
        println!("Generate Result: {:?}", result);
        println!("Message Complete: {:?}", message_complete.lock().await);
    }
    Err(e) => {
        eprintln!("Error calling generate: {:?}", e);
    }
}
```

### Agents

And agent and agent executor is a chain which can interact with `Tool` which are elemts of outside the llm it self, like search in google

##### Conversational Agent

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

async fn agent_run() {
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

}
```

### Chain

The Chain trait in LangChain Rust represents a powerful abstraction that allows developers to build sequences of operations, often involving language model interactions. This feature is crucial for creating sophisticated workflows that require a series of logical steps, such as generating text, processing information, and making decisions based on model outputs.

#### Conversational chain

Conversational chain keeps a memory of the chain, the prompt args should be input

```rust
let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
let chain = ConversationalChainBuilder::new()
    .llm(llm)
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
```

#### LLM Chain

- Default implementation

```rust
let human_message_prompt = HumanMessagePromptTemplate::new(template_fstring!(
    "Mi nombre es: {nombre} ",
    "nombre",
));

let formatter = message_formatter![MessageOrTemplate::Template(human_message_prompt.into()),];
let llm = OpenAI::default();
let chain = LLMChainBuilder::new()
    .prompt(formatter)
    .llm(llm)
    .build()
    .expect("Failed to build LLMChain");
let input_variables = prompt_args! {
    "nombre" => "luis",
};
match chain.invoke(input_variables).await {
    Ok(result) => {
        println!("Result: {:?}", result);
    }
    Err(e) => panic!("Error invoking LLMChain: {:?}", e),
}
```

- With stram and option implementation

```rust
let human_msg = Message::new_human_message("Hello from user");
// Create an AI message prompt template
let human_message_prompt = HumanMessagePromptTemplate::new(template_fstring!(
    "Mi nombre es: {nombre} ",
    "nombre",
));

let message_complete = Arc::new(Mutex::new(String::new()));

// Define the streaming function
// This function will append the content received from the stream to `message_complete`
let streaming_func = {
    let message_complete = message_complete.clone();
    move |content: String| {
        let message_complete = message_complete.clone();
        async move {
            let mut message_complete_lock = message_complete.lock().await;
            println!("Content: {:?}", content);
            message_complete_lock.push_str(&content);
            Ok(())
        }
    }
};
// Use the `message_formatter` macro to construct the formatter
let formatter = message_formatter![MessageOrTemplate::Template(human_message_prompt.into()),];

let options = ChainCallOptions::default().with_streaming_func(streaming_func);
let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
let chain =  LLMChainBuilder::new()
    .prompt(formatter)
    .llm(llm)
    .options(options)
    .build()
    .expect("Failed to build LLMChain");

let input_variables = prompt_args! {
    "nombre" => "luis",

};
match chain.invoke(input_variables).await {
    Ok(result) => {
        println!("Result: {:?}", result);
        println!("Complete message: {:?}", message_complete.lock().await);
    }
    Err(e) => panic!("Error invoking LLMChain: {:?}", e),
}
```

### Embeddings

```rust
use langchain::embedding::openai::OpenAiEmbedder;

let openai_embedder = OpenAiEmbedder::new("your_openai_api_key".to_string());
```

Or use the default implementation:

```rust
let openai_embedder = OpenAiEmbedder::default();
```

#### Embedding Documents

Embed multiple documents asynchronously:

```rust
#[tokio::main]
async fn main() {
    let documents = vec!["Hello, world!".to_string(), "How are you?".to_string()];
    let embeddings = openai_embedder.embed_documents(&documents).await.unwrap();

    println!("{:?}", embeddings);
}
```

### Embedding a Single Query

Embed a single piece of text:

```rust
#[tokio::main]
async fn main() {
    let query = "What is the meaning of life?";
    let embedding = openai_embedder.embed_query(query).await.unwrap();

    println!("{:?}", embedding);
}
```

### Messsages

#### Message Formatting

LangChain Rust provides a structured way to define different types of messages, such as system messages, AI-generated messages, and human messages. Below is an example of how to create and use these message types:

```rust
use serde::{Serialize, Deserialize};
use langchain::prompt::chat::{Message, MessageType};

// Creating different types of messages
let system_msg = Message::new_system_message("System initialization complete.");
let human_msg = Message::new_human_message("Hello, how can I assist you?");
let ai_msg = Message::new_ai_message("Analyzing input...");

println!("{:?}, {:?}, {:?}", system_msg, human_msg, ai_msg);
```

#### Dynamic Prompt Templates

LangChain Rust allows the creation of dynamic prompt templates using either FString or Jinja2 formatting. These templates can include variables that are replaced at runtime, as shown in the example below:

```rust
use langchain::prompt::{PromptTemplate, TemplateFormat, prompt_args, template_fstring};

// Creating a Jinja2 template for a greeting message
let greeting_template = template_fstring!(
    "Hello, {name}! Welcome to our service.",
    "name"
);

// Formatting the template with a given name
let formatted_greeting = greeting_template.format(prompt_args! {
    "name" => "Alice",
}).unwrap();

println!("{}", formatted_greeting);
```

#### Message Formatting with Templates

Combining message structures with dynamic templates, you can create complex conversational flows. The MessageFormatter allows the sequencing of various message types and templates, providing a cohesive conversation script:

```rust

let human_msg = Message::new_human_message("Hello from user");

// Create an AI message prompt template
let ai_message_prompt = AIMessagePromptTemplate::new(template_fstring!(
    "AI response: {content} {test}",
    "content",
    "test"
));

// Use the `message_formatter` macro to construct the formatter
let formatter = message_formatter![
    MessageOrTemplate::Message(human_msg),
    MessageOrTemplate::Template(ai_message_prompt.into()),
    MessageOrTemplate::MessagesPlaceholder("history".to_string())
];

// Define input variables for the AI message template
let input_variables = prompt_args! {
    "content" => "This is a test",
    "test" => "test2",
    "history" => vec![
        Message::new_human_message("Placeholder message 1"),
        Message::new_ai_message("Placeholder message 2"),
    ],


};

// Format messages
let formatted_messages = formatter.format_messages(input_variables).unwrap();

// Verify the number of messages
assert_eq!(formatted_messages.len(), 4);

// Verify the content of each message
assert_eq!(formatted_messages[0].content, "Hello from user");
assert_eq!(
    formatted_messages[1].content,
    "AI response: This is a test test2"
);
assert_eq!(formatted_messages[2].content, "Placeholder message 1");
assert_eq!(formatted_messages[3].content, "Placeholder message 2");
```

## License

`langchain-rust` is released under the MIT License. See the [LICENSE](https://github.com/gyroflaw/langchain_rs/blob/main/LICENSE) file for more information.
