// An APIChain answers questions with the Open-Meteo weather API (free, no key).
// The model writes the request from the docs below; the chain only calls api.open-meteo.com.
//
// OpenAI:            OPENAI_API_KEY=... cargo run --example api_chain
// Any local model:   API_CHAIN_OLLAMA_MODEL=qwen3:14b cargo run --example api_chain
//                    (talks to Ollama's OpenAI-compatible endpoint on localhost:11434)

use langchain_rust::{
    chain::{APIChain, Chain, API_CHAIN_REQUEST_KEY},
    llm::{openai::OpenAI, OpenAIConfig},
    prompt_args,
};

const OPEN_METEO_DOCS: &str = r#"Open-Meteo forecast API.

GET https://api.open-meteo.com/v1/forecast
Query parameters:
- latitude, longitude (required): the location, in decimal degrees.
- current: comma-separated variables for the current weather, e.g.
  temperature_2m, relative_humidity_2m, wind_speed_10m, precipitation, weather_code.
- timezone: e.g. auto.
Returns JSON with a "current" object holding the requested variables and "current_units"
with their units."#;

#[tokio::main]
async fn main() {
    let llm = match std::env::var("API_CHAIN_OLLAMA_MODEL") {
        Ok(model) => OpenAI::default()
            .with_config(
                OpenAIConfig::default()
                    .with_api_base("http://localhost:11434/v1")
                    .with_api_key("ollama"),
            )
            .with_model(model),
        Err(_) => OpenAI::default(),
    };

    let chain = APIChain::new(
        llm,
        OPEN_METEO_DOCS,
        ["https://api.open-meteo.com/v1/forecast"],
    )
    .unwrap();

    let question = "What's the temperature and wind speed in Lima, Peru right now?";
    let output = chain
        .execute(prompt_args! {"input" => question})
        .await
        .unwrap();
    println!("request: {}", output[API_CHAIN_REQUEST_KEY]);
    println!("answer: {}", output["answer"]);
}
