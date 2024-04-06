use langchain_rust::{language_models::llm::LLM, llm::Claude};

#[tokio::main]
async fn main() {
    let claude = Claude::default().with_model("claude-3-opus-20240229");
    let response = claude.invoke("hola").await.unwrap();
    println!("{}", response);
}
