#[cfg(feature = "fastembed")]
use langchain_rust::embedding::{Embedder, EmbeddingModel, FastEmbed, InitOptions, TextEmbedding};

#[cfg(feature = "fastembed")]
#[tokio::main]
async fn main() {
    // With default model
    let fastembed = FastEmbed::try_new().unwrap();
    let embeddings = fastembed
        .embed_documents(&["hello world".to_string(), "foo bar".to_string()])
        .await
        .unwrap();

    println!("Len: {}", embeddings.len());
    println!("Embeddings: {:?}", embeddings);

    // With custom model
    let model = TextEmbedding::try_new(InitOptions {
        model_name: EmbeddingModel::AllMiniLML6V2,
        show_download_progress: true,
        ..Default::default()
    })
    .unwrap();

    let fastembed = FastEmbed::from(model);

    fastembed
        .embed_documents(&["hello world".to_string(), "foo bar".to_string()])
        .await
        .unwrap();

    println!("Len: {:?}", embeddings.len());
}

#[cfg(not(feature = "fastembed"))]
fn main() {
    println!("This example requires the 'fastembed' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example embedding_fastembed --features=fastembed");
}
