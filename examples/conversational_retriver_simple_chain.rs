use std::error::Error;

use async_trait::async_trait;
use futures_util::StreamExt;
use langchain_rust::{
    chain::{Chain, ConversationalRetriverChainBuilder},
    llm::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
    prompt_args,
    schemas::{Document, Retriever},
};

struct RetriverMock {}
#[async_trait]
impl Retriever for RetriverMock {
    async fn get_relevant_documents(
        &self,
        _question: &str,
    ) -> Result<Vec<Document>, Box<dyn Error>> {
        Ok(vec![
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
        ])
    }
}
#[tokio::main]
async fn main() {
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt35.to_string());
    let chain = ConversationalRetriverChainBuilder::new()
        .llm(llm)
        .rephrase_question(true)
        .retriver(RetriverMock {})
        .memory(SimpleMemory::new().into())
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
