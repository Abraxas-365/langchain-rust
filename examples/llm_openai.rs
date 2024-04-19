use langchain_rust::{language_models::llm::LLM, llm::openai::OpenAI};

#[tokio::main]
async fn main() {
    //OpenAI Example
    let open_ai = OpenAI::default();
    let response = open_ai.invoke("hola").await.unwrap();
    println!("{}", response);

    //or whe can set config as
    let open_ai = OpenAI::default().with_config(
        OpenAIConfig::default()
            .with_api_base("xxx") //if you want to specify base url
            .with_api_key("<you_api_key>"), //if you want to set you open ai key,
    );

    let response = ollama.invoke("hola").await.unwrap();
    println!("{}", response);
}
