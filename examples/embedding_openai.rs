use langchain_rust::embedding::{embedder_trait::Embedder, openai::OpenAiEmbedder};

#[tokio::main]
async fn main() {
    let openai = OpenAiEmbedder::default();

    let response = openai.embed_query("What is the sky blue?").await.unwrap();

    println!("{:?}", response);
}
