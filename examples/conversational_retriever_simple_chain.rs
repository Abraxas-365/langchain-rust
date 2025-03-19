use async_trait::async_trait;
use futures_util::StreamExt;
use indoc::indoc;
use langchain_rust::{
    chain::{Chain, ConversationalRetrieverChainBuilder, StuffQABuilder},
    llm::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
    prompt_template,
    schemas::{Document, Message, MessageType, Retriever},
    template::MessageTemplate,
};
use std::error::Error;

struct RetrieverMock {}
#[async_trait]
impl Retriever for RetrieverMock {
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
                "Whats his favorite food", "Pan con chicharron"
            )),
        ])
    }
}
#[tokio::main]
async fn main() {
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt35.to_string());
    let prompt = prompt_template![
        Message::new(MessageType::SystemMessage, "You are a helpful assistant"),
        MessageTemplate::from_jinja2(
            MessageType::HumanMessage,
            indoc! {"
            Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {{context}}

            Question:{{question}}
            Helpful Answer:

            "},
        )
    ];
    let chain = ConversationalRetrieverChainBuilder::new()
        .llm(llm)
        .rephrase_question(true)
        .retriever(RetrieverMock {})
        .memory(SimpleMemory::new().into())
        //If you want to use the default prompt remove the .prompt()
        //Keep in mind if you want to change the prompt; this chain need the {{context}} variable
        .prompt(prompt)
        .build()
        .expect("Error building ConversationalChain");

    let mut input_variables = StuffQABuilder::new().question("Hi").build();

    let result = chain.invoke(&mut input_variables).await;
    if let Ok(result) = result {
        println!("Result: {:?}", result);
    }

    let mut input_variables = StuffQABuilder::new()
        .question("Which is luis Favorite Food")
        .build();

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
