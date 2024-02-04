# langchain-rust

## Embeddings

### Usage

#### Setup

Initialize components such as the OpenAiEmbedder for embeddings:

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
