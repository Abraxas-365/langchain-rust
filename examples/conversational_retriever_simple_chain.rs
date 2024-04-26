use std::error::Error;

use async_trait::async_trait;
use futures_util::StreamExt;
use langchain_rust::{
    chain::{Chain, ConversationalRetrieverChainBuilder},
    fmt_message, fmt_template,
    llm::{OpenAI, OpenAIModel},
    memory::SimpleMemory,
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::{Document, Message, Retriever},
    template_jinja2,
};

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
                "Whts his favorite food", "Pan con chicharron"
            )),
        ])
    }
}
#[tokio::main]
async fn main() {
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt35.to_string());
    let prompt=message_formatter![
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
        .retriever(RetrieverMock {})
        .memory(SimpleMemory::new().into())
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
