use langchain_rust::{
    chain::{Chain, StuffDocumentBuilder, StuffQABuilder},
    llm::openai::OpenAI,
    schemas::Document,
};

#[tokio::main]
async fn main() {
    let llm = OpenAI::default();

    let chain = StuffDocumentBuilder::new()
        .llm(llm)
        // .prompt() you can add a custom prompt if you want
        .build()
        .unwrap();
    let mut input = StuffQABuilder::new()
        .question("How old is luis and whats his favorite text editor")
        .documents(&[
            Document::new(format!(
                "\nQuestion: {}\nAnswer: {}\n",
                "Which is the favorite text editor of luis", "Nvim"
            )),
            Document::new(format!(
                "\nQuestion: {}\nAnswer: {}\n",
                "How old is Luis", "24"
            )),
        ])
        .build();

    let output = chain.invoke(&mut input).await.unwrap();

    println!("{}", output);
}
