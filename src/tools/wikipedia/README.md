# Wikipedia Tool for LangChain Rust

A powerful and flexible Wikipedia API tool for LangChain Rust, enabling seamless integration of Wikipedia knowledge into your LLM applications.

## Features

- 🔍 **Smart Search**: Search Wikipedia articles with configurable result limits
- 🌐 **Multi-language Support**: Query Wikipedia in any language (English, Spanish, French, German, etc.)
- ⚙️ **Highly Configurable**: Customize search results, content length, and language
- 🚀 **Async/Await**: Built with Tokio for efficient async operations
- 🧪 **Well Tested**: Comprehensive unit and integration tests
- 🤖 **Agent Ready**: Seamlessly integrates with LangChain agents

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
langchain-rust = "4.6"
tokio = { version = "1.0", features = ["full"] }
serde_json = "1.0"
```

## Quick Start

### Basic Usage

```rust
use langchain_rust::tools::{Tool, WikipediaQuery};
use serde_json::json;

#[tokio::main]
async fn main() {
    let wiki = WikipediaQuery::default();
    let result = wiki.run(json!("Rust programming language")).await.unwrap();
    println!("{}", result);
}
```

### Custom Configuration

```rust
use langchain_rust::tools::{WikipediaQuery, WikipediaQueryOptions};
use serde_json::json;

#[tokio::main]
async fn main() {
    let options = WikipediaQueryOptions {
        top_k_results: 5,
        max_doc_content_length: 2000,
        lang: "en".to_string(),
    };
    
    let wiki = WikipediaQuery::new(options);
    let result = wiki.run(json!("Artificial Intelligence")).await.unwrap();
    println!("{}", result);
}
```

### Builder Pattern

```rust
use langchain_rust::tools::{Tool, WikipediaQuery};
use serde_json::json;

#[tokio::main]
async fn main() {
    let wiki = WikipediaQuery::default()
        .with_top_k_results(3)
        .with_max_doc_content_length(1500);
    
    let result = wiki.run(json!("Machine Learning")).await.unwrap();
    println!("{}", result);
}
```

## Advanced Usage

### Multi-language Support

Query Wikipedia in different languages:

```rust
use langchain_rust::tools::WikipediaQuery;
use serde_json::json;

#[tokio::main]
async fn main() {
    // Spanish Wikipedia
    let wiki_es = WikipediaQuery::with_lang("es");
    let result = wiki_es.run(json!("Inteligencia artificial")).await.unwrap();
    
    // French Wikipedia
    let wiki_fr = WikipediaQuery::with_lang("fr");
    let result = wiki_fr.run(json!("Intelligence artificielle")).await.unwrap();
    
    // German Wikipedia
    let wiki_de = WikipediaQuery::with_lang("de");
    let result = wiki_de.run(json!("Künstliche Intelligenz")).await.unwrap();
}
```

### Integration with Agents

Use Wikipedia tool with LangChain agents:

```rust
use langchain_rust::{
    agent::{AgentExecutor, OpenAiToolAgentBuilder},
    chain::Chain,
    llm::openai::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
    prompt_args,
    tools::WikipediaQuery,
};
use std::sync::Arc;

#[tokio::main]
async fn main() {
    // Initialize LLM
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt4oMini.to_string());
    
    // Create Wikipedia tool
    let wiki = Arc::new(WikipediaQuery::default());
    
    // Build agent with Wikipedia tool
    let agent = OpenAiToolAgentBuilder::new()
        .tools(vec![wiki])
        .build(llm)
        .unwrap();
    
    // Create executor with memory
    let executor = AgentExecutor::from_agent(agent)
        .with_memory(SimpleMemory::new().into());
    
    // Ask questions
    let result = executor.invoke(prompt_args! {
        "input" => "Tell me about the history of Rust programming language"
    }).await.unwrap();
    
    println!("{:?}", result);
}
```

## Configuration Options

### WikipediaQueryOptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `top_k_results` | `usize` | `3` | Maximum number of search results to return |
| `max_doc_content_length` | `usize` | `4000` | Maximum length of each document content in characters |
| `lang` | `String` | `"en"` | Language code for Wikipedia (e.g., "en", "es", "fr") |

## API Reference

### WikipediaQuery

Main struct implementing the `Tool` trait.

#### Methods

- `new(options: WikipediaQueryOptions) -> Self` - Creates a new instance with custom options
- `default() -> Self` - Creates a new instance with default options
- `with_lang(lang: impl Into<String>) -> Self` - Creates an instance for a specific language
- `with_top_k_results(top_k: usize) -> Self` - Sets the maximum number of results
- `with_max_doc_content_length(max_len: usize) -> Self` - Sets the maximum content length

#### Tool Trait Implementation

- `name() -> String` - Returns "wikipedia-api"
- `description() -> String` - Returns tool description
- `run(input: Value) -> Result<String, Box<dyn Error>>` - Executes the Wikipedia query

## Input Formats

The tool accepts input in two formats:

### String Input
```rust
wiki.run(json!("Rust programming language")).await
```

### Object Input
```rust
wiki.run(json!({"input": "Rust programming language"})).await
```

## Output Format

The tool returns results in the following format:

```
Page: Rust (programming language)
Summary: Rust is a multi-paradigm, general-purpose programming language...

Page: Rust
Summary: Rust is an iron oxide, a usually reddish-brown oxide...
```

## Error Handling

The tool handles various error scenarios:

```rust
use langchain_rust::tools::{Tool, WikipediaQuery};
use serde_json::json;

#[tokio::main]
async fn main() {
    let wiki = WikipediaQuery::default();
    
    // Handle empty queries
    match wiki.run(json!("")).await {
        Ok(_) => println!("Success"),
        Err(e) => eprintln!("Error: {}", e),
    }
    
    // Handle network errors
    match wiki.run(json!("Some query")).await {
        Ok(result) => println!("{}", result),
        Err(e) => eprintln!("Network error: {}", e),
    }
}
```

## Testing

### Running Tests

```bash
# Run unit tests
cargo test

# Run integration tests (requires internet)
cargo test -- --ignored

# Run specific test
cargo test test_basic_usage

# Run with output
cargo test -- --nocapture
```

### Test Coverage

The tool includes:
- Unit tests for configuration and setup
- Integration tests for API interactions
- Error handling tests
- Multi-language tests
- Content truncation tests

## Supported Languages

The tool supports all Wikipedia language editions. Common language codes:

- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `ru` - Russian
- `ja` - Japanese
- `zh` - Chinese
- `ar` - Arabic

For a complete list, see [Wikipedia Language Codes](https://en.wikipedia.org/wiki/List_of_Wikipedias).

## Performance Considerations

- **Rate Limiting**: The Wikipedia API has rate limits. Consider implementing caching for production use.
- **Content Size**: Large results are automatically truncated based on `max_doc_content_length`.
- **Network Latency**: Queries are async and non-blocking.
- **Concurrent Requests**: The tool uses `reqwest` with connection pooling for efficiency.

## Best Practices

1. **Set Appropriate Limits**: Configure `top_k_results` and `max_doc_content_length` based on your needs
2. **Handle Errors Gracefully**: Always wrap tool calls in proper error handling
3. **Use Caching**: Consider caching frequently accessed pages
4. **Respect Rate Limits**: Don't make excessive concurrent requests
5. **Choose the Right Language**: Use the appropriate language code for your use case

## Troubleshooting

### Common Issues

**Issue**: "Query cannot be empty"
- **Solution**: Ensure your input is not empty or whitespace-only

**Issue**: Network timeout
- **Solution**: Check your internet connection or increase timeout settings

**Issue**: No results found
- **Solution**: Try different search terms or check the language setting

**Issue**: Content truncated
- **Solution**: Increase `max_doc_content_length` in options

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Write tests for new features
2. Update documentation
3. Follow Rust best practices
4. Ensure all tests pass before submitting

## License

This tool is part of the LangChain Rust project and follows the same license.

## Acknowledgments

- Based on the LangChain.js Wikipedia tool implementation
- Uses the MediaWiki API
- Built with Rust's async ecosystem (Tokio, Reqwest)

## See Also

- [LangChain Rust Documentation](https://github.com/Abraxas-365/langchain-rust)
- [MediaWiki API Documentation](https://www.mediawiki.org/wiki/API:Main_page)
- [Wikipedia API Tutorial](https://www.mediawiki.org/wiki/API:Tutorial)

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/Abraxas-365/langchain-rust).
