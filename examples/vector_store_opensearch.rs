// To run this example execute: cargo run --example vector_store_opensearch --features opensearch

use async_openai::config::AzureConfig;
use aws_config::BehaviorVersion;
use langchain_rust::vectorstore::{VecStoreOptions, VectorStore};
#[cfg(feature = "opensearch")]
use langchain_rust::{
    embedding::openai::openai_embedder::OpenAiEmbedder, schemas::Document,
    vectorstore::opensearch::Store, vectorstore::opensearch::*,
};
use serde_json::json;
use std::collections::HashMap;
use std::error::Error;
use std::io::Write;

#[cfg(feature = "opensearch")]
#[tokio::main]
async fn main() {
    // In this example we use azure openai
    let api_base = "https://blahblah-aoai-fc.openai.azure.com";
    let api_key = "blahblah";
    let api_version = "2024-02-01";
    let embedding_model = "text-embedding-ada-002";

    let openai_config = AzureConfig::new()
        .with_api_base(api_base)
        .with_api_key(api_key)
        .with_api_version(api_version)
        .with_deployment_id(embedding_model);
    let embedder = OpenAiEmbedder::new(openai_config);

    // In this example we use an AWS profile to get (temporary credentials)
    std::env::set_var("AWS_PROFILE", "xxx");
    let sdk_config = aws_config::load_defaults(BehaviorVersion::latest()).await;

    // AOSS configuration
    let index = "test_aoss_2";
    let aoss_host = "https://blahblah.eu-central-1.aoss.amazonaws.com/";

    let store = StoreBuilder::new()
        .embedder(embedder)
        .index(index)
        .content_field("the_content_field")
        .vector_field("the_vector_field")
        .aoss_client(&sdk_config, aoss_host)
        .unwrap()
        .build()
        .await
        .unwrap();

    //store.delete_index().await.unwrap();
    //store.create_index().await.unwrap();
    //let added_ids = add_documents_to_index(&store).await.unwrap();
    //for id in added_ids {
    //    println!("added {id}");
    //}
    // it can take a while before the documents are actually available in the index...

    // Ask for user input
    print!("Query> ");
    std::io::stdout().flush().unwrap();
    let mut query = String::new();
    std::io::stdin().read_line(&mut query).unwrap();

    let results = store
        .similarity_search(&query, 2, &VecStoreOptions::default())
        .await
        .unwrap();

    if results.is_empty() {
        println!("No results found.");
        return;
    } else {
        results.iter().for_each(|r| {
            println!("Document: {}", r.page_content);
        });
    }
}

async fn add_documents_to_index(store: &Store) -> Result<Vec<String>, Box<dyn Error>> {
    let doc1 = Document::new(
        "langchain-rust is a port of the langchain python library to rust and was written in 2024.",
    )
    .with_metadata(HashMap::from([("source".to_string(), json!("cli"))]));

    let doc2 = Document::new(
        "langchaingo is a port of the langchain python library to go language and was written in 2023."
    );

    let doc3 = Document::new(
        "Capital of United States of America (USA) is Washington D.C. and the capital of France is Paris."
    );

    let doc4 = Document::new("Capital of France is Paris.");

    let opts = VecStoreOptions {
        name_space: None,
        score_threshold: None,
        filters: None,
        embedder: Some(store.embedder.clone()),
    };

    let result = store
        .add_documents(&vec![doc1, doc2, doc3, doc4], &opts)
        .await?;

    Ok(result)
}

#[cfg(not(feature = "opensearch"))]
fn main() {
    println!("This example requires the 'opensearch' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example vector_store_opensearch --features opensearch");
}
