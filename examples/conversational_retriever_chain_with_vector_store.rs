// To run this example execute: cargo run --example conversational_retriever_chain --features postgres

#[cfg(feature = "postgres")]
use futures_util::StreamExt;
#[cfg(feature = "postgres")]
use langchain_rust::{
    add_documents,
    chain::{Chain, ConversationalRetrieverChainBuilder},
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
    use langchain_rust::{
        fmt_message, fmt_template, message_formatter, prompt::HumanMessagePromptTemplate,
        schemas::Message, template_jinja2,
    };

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
    let prompt= message_formatter![
                    fmt_message!(Message::new_system_message("You are a helpful assistant")),
                    fmt_template!(HumanMessagePromptTemplate::new(
                    template_jinja2!("
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{{context}}

Question:{{question}}
Helpful Answer:

        ",
                    "context","question")))

                ];

    let chain = ConversationalRetrieverChainBuilder::new()
        .llm(llm)
        .rephrase_question(true)
        .memory(SimpleMemory::new().into())
        .retriever(Retriever::new(store, 5))
        //If you want to sue the default prompt remove the .prompt()
        //Keep in mind if you want to change the prmpt; this chain need the {{context}} variable
        .prompt(prompt)
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
    println!("cargo run --example conversational_retriever_chain --features postgres");
}
