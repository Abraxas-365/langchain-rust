// To run this example execute: cargo run --example vector_store_opensearch --features opensearch

use langchain_rust::vectorstore::{VecStoreOptions, VectorStore};
#[cfg(feature = "opensearch")]
use langchain_rust::{
    embedding::openai::openai_embedder::OpenAiEmbedder, schemas::Document,
    vectorstore::opensearch::Store, vectorstore::opensearch::*,
};
use opensearch::auth::Credentials;
use opensearch::cert::CertificateValidation;
use opensearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use opensearch::OpenSearch;
use serde_json::json;
use std::collections::HashMap;
use std::error::Error;
use std::io::Write;
use aws_config::SdkConfig;
use url::Url;

#[cfg(feature = "opensearch")]
#[tokio::main]
async fn main() {
    // Initialize Embedder
    let embedder = OpenAiEmbedder::default();

    /* In this example we use an opensearch instance running on localhost (docker):

    docker run --rm -it -p 9200:9200 -p 9600:9600 \
        -e "discovery.type=single-node" \
        -e "node.name=localhost" \
        -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=ZxPZ3cZL0ky1bzVu+~N" \
        opensearchproject/opensearch:latest

     */

    let opensearch_host = "https://localhost:9200";
    let opensearch_index = "test";

    let url = Url::parse(opensearch_host).unwrap();
    let conn_pool = SingleNodeConnectionPool::new(url);
    let transport = TransportBuilder::new(conn_pool)
        .disable_proxy()
        .auth(Credentials::Basic(
            "admin".to_string(),
            "ZxPZ3cZL0ky1bzVu+~N".to_string(),
        ))
        .cert_validation(CertificateValidation::None)
        .build()
        .unwrap();
    let client = OpenSearch::new(transport);

    // We could also use an AOSS instance:
    // std::env::set_var("AWS_PROFILE", "xxx");
    // use aws_config::BehaviorVersion;
    // let sdk_config = aws_config::load_defaults(BehaviorVersion::latest()).await;
    // let aoss_host = "https://blahblah.eu-central-1.aoss.amazonaws.com/";
    // let client = build_aoss_client(&sdk_config, aoss_host).unwrap();

    let store = StoreBuilder::new()
        .embedder(embedder)
        .index(opensearch_index)
        .content_field("the_content_field")
        .vector_field("the_vector_field")
        .client(client)
        .build()
        .await
        .unwrap();

    let _ = store.delete_index().await;
    store.create_index().await.unwrap();
    let added_ids = add_documents_to_index(&store).await.unwrap();
    for id in added_ids {
        println!("added document with id: {id}");
    }
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

#[allow(dead_code)]
fn build_aoss_client(sdk_config: &SdkConfig, host: &str) -> Result<OpenSearch, Box<dyn Error>> {
    let opensearch_url = Url::parse(host)?;
    let conn_pool = SingleNodeConnectionPool::new(opensearch_url);

    let transport = TransportBuilder::new(conn_pool)
        .auth(sdk_config.try_into()?)
        .service_name("aoss")
        .build()?;
    let client = OpenSearch::new(transport);
    Ok(client)
}

#[cfg(not(feature = "opensearch"))]
fn main() {
    println!("This example requires the 'opensearch' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example vector_store_opensearch --features opensearch");
}
