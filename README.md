# 🦜️🔗LangChain Rust

⚡ Building applications with LLMs through composability, with Rust! ⚡

## 🤔 What is this?

This is the Rust language implementation of [LangChain](https://github.com/langchain-ai/langchain).

## Examples

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
use langchain::prompt::chat::{Message, messages_placeholder, message_formatter, MessageOrTemplate};
use langchain::prompt::{template_fstring, prompt_args};
use std::collections::HashMap;

// Define a formatter using the `message_formatter` macro to construct a conversation flow
let formatter = message_formatter![
    // Add a human message directly
    MessageOrTemplate::Message(Message::new_human_message("User query: What's the weather like?")),

    // Add an AI message using a template
    MessageOrTemplate::Template(template_fstring!(
        "AI response: The current weather is {weather_condition}.",
        "weather_condition"
    )),

    // Add a placeholder for additional messages using `messages_placeholder` macro
    messages_placeholder![
        Message::new_system_message("Processing your request..."),
        Message::new_system_message("End of conversation."),
    ],
];

// Define input variables for the AI message template
let input_variables = prompt_args! {
    "weather_condition" => "sunny with a chance of rain",
};

// Format the conversation using the formatter and input variables
let conversation_script = formatter.format(input_variables).unwrap();

// Print out each message in the formatted conversation
for message in conversation_script {
    println!("{:?}", message.content);
}
```
