// To run this example execute: cargo run --example conversational_retriever_chain --features postgres

#[cfg(feature = "postgres")]
use futures_util::StreamExt;
#[cfg(feature = "postgres")]
use indoc::indoc;
#[cfg(feature = "postgres")]
use langchain_rust::{
    add_documents,
    chain::StuffQABuilder,
    chain::{Chain, ConversationalRetrieverChainBuilder},
    embedding::openai::openai_embedder::OpenAiEmbedder,
    llm::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
    prompt_template,
    schemas::Document,
    schemas::{Message, MessageType},
    template::MessageTemplate,
    vectorstore::{pgvector::StoreBuilder, Retriever, VectorStore},
};

#[cfg(feature = "postgres")]
#[tokio::main]
async fn main() {
    use langchain_rust::schemas::InputVariables;

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
            "Whats his favorite food", "Pan con chicharron"
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
    let prompt = prompt_template![
        Message::new(MessageType::SystemMessage, "You are a helpful assistant"),
        MessageTemplate::from_jinja2(
            MessageType::HumanMessage,
            indoc! {"
                Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

                {{context}}

                Question: {{question}}
                Helpful Answer:

            "},
        )
    ];

    let chain = ConversationalRetrieverChainBuilder::new()
        .llm(llm)
        .rephrase_question(true)
        .memory(SimpleMemory::new().into())
        .retriever(Retriever::new(store, 5))
        //If you want to use the default prompt remove the .prompt()
        //Keep in mind if you want to change the prompt; this chain need the {{context}} variable
        .prompt(prompt)
        .build()
        .expect("Error building ConversationalChain");

    let mut input_variables: InputVariables = StuffQABuilder::new().question("Hi").build().into();

    let result = chain.invoke(&mut input_variables).await;
    if let Ok(result) = result {
        println!("Result: {:?}", result);
    }

    let mut input_variables: InputVariables = StuffQABuilder::new()
        .question("Which is luis Favorite Food")
        .build()
        .into();

    //If you want to stream
    let mut stream = chain.stream(&mut input_variables).await.unwrap();
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
    println!("cargo run --example conversational_retriever_chain --features postgres");
}
