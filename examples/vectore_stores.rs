//To run this example ecute: cargo run --example vectore_stores --features postgres
use langchain_rust::{
    add_documents,
    embedding::openai::openai_embedder::OpenAiEmbedder,
    schemas::Document,
    similarity_search,
    vectorstore::{pgvector::StoreBuilder, VectorStore},
};
use std::io::Write;
use tokio::io::{self, AsyncBufReadExt, BufReader};

#[tokio::main]
async fn main() {
    //Inicializo el ebedder
    let embedder = OpenAiEmbedder::default();

    //Inicializo el Vectore Store de postgres
    let store = StoreBuilder::new()
        .embedder(embedder)
        .pre_delete_collection(true)
        .connection_url("postgresql://postgres:postgres@localhost:5432/postgres")
        .vector_dimensions(1536)
        .build()
        .await
        .unwrap();

    //Obtiene el input con lista de palabras
    let mut input = String::new();
    print!("Please, enter a list separated by commas: ");
    std::io::stdout().flush().unwrap();
    let mut reader = BufReader::new(io::stdin());
    reader.read_line(&mut input).await.unwrap();
    let input = input.trim_end();
    let list: Vec<&str> = input.split(',').collect();

    //Lo vulevo una lista de documentos
    let documents: Vec<Document> = list
        .iter()
        .map(|text| Document::new(text.trim().to_string()))
        .collect();

    //Anado los documetos a la base de datos
    let _ = add_documents!(store, &documents).await.map_err(|e| {
        println!("Error adding documents: {:?}", e);
    });

    //Se obtiene el input que buscar
    let mut search_input = String::new();
    print!("Please enter the text you want to search: ");
    std::io::stdout().flush().unwrap();

    reader.read_line(&mut search_input).await.unwrap();
    let search_input = search_input.trim_end();

    //Se hace un similarity search en la base de datos
    let data = similarity_search!(store, search_input, 10)
        .await
        .map_err(|e| {
            println!("Error searching documents: {:?}", e);
        })
        .unwrap();

    data.iter().for_each(|d| println!("{:?}", d.page_content));
}
