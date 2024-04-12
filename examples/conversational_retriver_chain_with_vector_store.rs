// To run this example execute: cargo run --example conversational_retriver_chain --features postgres

#[cfg(feature = "postgres")]
use futures_util::StreamExt;
#[cfg(feature = "postgres")]
use langchain_rust::{
    add_documents,
    chain::{Chain, ConversationalRetriverChainBuilder},
    embedding::openai::openai_embedder::OpenAiEmbedder,
    llm::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
    prompt_args,
    schemas::Document,
    vectorstore::{pgvector::StoreBuilder, Retriever, VectorStore},
};

#[cfg(feature = "postgres")]
#[tokio::main]
async fn main() {
    let documents = vec![
        Document::new(format!(
            "\nQuestion: {}\nAnswer: {}\n",
            "Which is the favorite text editor of luis", "Nvim"
        )),
        Document::new(format!(
            "\nQuestion: {}\nAnswer: {}\n",
            "How old is Luis", "24"
        )),
        Document::new(format!(
            "\nQuestion: {}\nAnswer: {}\n",
            "Where do luis live", "Peru"
        )),
        Document::new(format!(
            "\nQuestion: {}\nAnswer: {}\n",
            "Whts his favorite food", "Pan con chicharron"
        )),
    ];

    let store = StoreBuilder::new()
        .embedder(OpenAiEmbedder::default())
        .pre_delete_collection(true)
        .connection_url("postgresql://postgres:postgres@localhost:5432/postgres")
        .vector_dimensions(1536)
        .build()
        .await
        .unwrap();

    let _ = add_documents!(store, &documents).await.map_err(|e| {
        println!("Error adding documents: {:?}", e);
    });

    let llm = OpenAI::default().with_model(OpenAIModel::Gpt35.to_string());

    let chain = ConversationalRetriverChainBuilder::new()
        .llm(llm)
        .memory(SimpleMemory::new().into())
        .retriver(Retriever::new(store, 5))
        .build()
        .expect("Error building ConversationalChain");

    let input_variables = prompt_args! {
        "question" => "Hi",
    };

    let result = chain.invoke(input_variables).await;
    if let Ok(result) = result {
        println!("Result: {:?}", result);
    }

    let input_variables = prompt_args! {
        "question" => "Which is luis Favorite Food",
    };

    //If you want to stream
    let mut stream = chain.stream(input_variables).await.unwrap();
    while let Some(result) = stream.next().await {
        match result {
            Ok(data) => data.to_stdout().unwrap(),
            Err(e) => {
                println!("Error: {:?}", e);
            }
        }
    }
}

#[cfg(not(feature = "postgres"))]
fn main() {
    println!("This example requires the 'postgres' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example conversational_retriver_chain --features postgres");
}
