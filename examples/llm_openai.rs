use langchain_rust::{language_models::llm::LLM, llm::openai::OpenAI};

#[tokio::main]
async fn main() {
    //OpenAI Example
    let open_ai = OpenAI::default();
    let response = open_ai.invoke("hola").await.unwrap();
    println!("{}", response);
}
