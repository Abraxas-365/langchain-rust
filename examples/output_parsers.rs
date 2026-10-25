// Typed output parsers: tell the model what format to answer in, then parse the answer
// into Rust values.
//
// OpenAI:            OPENAI_API_KEY=... cargo run --example output_parsers
// Any local model:   PARSERS_OLLAMA_MODEL=qwen3:14b cargo run --example output_parsers
//                    (talks to Ollama's OpenAI-compatible endpoint on localhost:11434)

use langchain_rust::{
    language_models::llm::LLM,
    llm::{openai::OpenAI, OpenAIConfig},
    output_parsers::{
        BooleanParser, CombiningParser, CommaSeparatedListParser, JsonParser, RegexDictParser,
        ResponseSchema, StructuredOutputParser, TypedOutputParser,
    },
};
use serde::Deserialize;
use serde_json::json;

#[derive(Debug, Deserialize)]
struct Movie {
    title: String,
    year: u16,
    director: String,
    genres: Vec<String>,
}

#[tokio::main]
async fn main() {
    let llm = match std::env::var("PARSERS_OLLAMA_MODEL") {
        Ok(model) => OpenAI::default()
            .with_config(
                OpenAIConfig::default()
                    .with_api_base("http://localhost:11434/v1")
                    .with_api_key("ollama"),
            )
            .with_model(model),
        Err(_) => OpenAI::default(),
    };

    // A typed struct straight from the model's JSON.
    let movie_parser = JsonParser::<Movie>::new().with_schema(json!({
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "year": {"type": "integer"},
            "director": {"type": "string"},
            "genres": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["title", "year", "director", "genres"]
    }));
    let answer = llm
        .invoke(&format!(
            "Tell me about the first Alien movie.\n\n{}",
            movie_parser.format_instructions()
        ))
        .await
        .unwrap();
    let movie: Movie = movie_parser.parse_output(&answer).unwrap();
    println!(
        "{} ({}), directed by {}: {}",
        movie.title,
        movie.year,
        movie.director,
        movie.genres.join(", ")
    );

    // A list.
    let list_parser = CommaSeparatedListParser::new();
    let answer = llm
        .invoke(&format!(
            "Name five ice cream flavors.\n{}",
            list_parser.format_instructions()
        ))
        .await
        .unwrap();
    println!("flavors: {:?}", list_parser.parse_output(&answer).unwrap());

    // A yes/no decision.
    let yes_no = BooleanParser::new();
    let answer = llm
        .invoke(&format!(
            "Is Lima the capital of Peru? {}",
            yes_no.format_instructions()
        ))
        .await
        .unwrap();
    println!(
        "Lima is the capital: {}",
        yes_no.parse_output(&answer).unwrap()
    );

    // Named fields without defining a type.
    let structured = StructuredOutputParser::new([
        ResponseSchema::new("answer", "answer to the user's question"),
        ResponseSchema::new("confidence", "how sure you are, from 0 to 1"),
    ]);
    let answer = llm
        .invoke(&format!(
            "What is the boiling point of water at sea level in Celsius?\n\n{}",
            structured.format_instructions()
        ))
        .await
        .unwrap();
    let fields = structured.parse_output(&answer).unwrap();
    println!(
        "answer: {}, confidence: {}",
        fields["answer"], fields["confidence"]
    );

    // Two formats in one answer: a JSON block, then labelled lines.
    let combined = CombiningParser::new(
        StructuredOutputParser::new([
            ResponseSchema::new("city", "the capital city"),
            ResponseSchema::new("country", "the country it is the capital of"),
        ]),
        RegexDictParser::new([("population", "Population"), ("source", "Source")])
            .with_no_update_value("N/A"),
    );
    let answer = llm
        .invoke(&format!(
            "What is the capital of Japan, and roughly how many people live there?\n\n{}",
            combined.format_instructions()
        ))
        .await
        .unwrap();
    let fields = combined.parse_output(&answer).unwrap();
    println!("combined: {}", serde_json::Value::Object(fields));
}
