#[cfg(feature = "mistralai")]
use langchain_rust::embedding::{embedder_trait::Embedder, mistralai::MistralAIEmbedder};

#[cfg(feature = "mistralai")]
#[tokio::main]
async fn main() {
    let mistralai = MistralAIEmbedder::try_new().unwrap();

    let embedding = mistralai.embed_query("Why is the sky blue?").await.unwrap();

    println!("{:?}", embedding);
}

#[cfg(not(feature = "mistralai"))]
fn main() {
    println!("This example requires the 'mistralai' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example embedding_mistralai --features=mistralai");
}
