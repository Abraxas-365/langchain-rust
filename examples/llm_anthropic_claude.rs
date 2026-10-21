use langchain_rust::{
    language_models::llm::LLM,
    llm::{Claude, ClaudeModel},
};

#[tokio::main]
async fn main() {
    let claude = Claude::default().with_model(ClaudeModel::ClaudeSonnet5_5);
    let response = claude.invoke("hola").await.unwrap();
    println!("{}", response);
}
