use langchain_rust::{
    chain::{Chain, StuffDocumentBuilder, StuffQA},
    llm::openai::OpenAI,
    prompt_args,
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
    let mut input = StuffQA::new(
        vec![
            Document::new(format!(
                "\nQuestion: {}\nAnswer: {}\n",
                "Which is the favorite text editor of luis", "Nvim"
            )),
            Document::new(format!(
                "\nQuestion: {}\nAnswer: {}\n",
                "How old is Luis", "24"
            )),
        ],
        prompt_args! {
            "question"=>"How old is luis and whats his favorite text editor"
        },
    );

    let output = chain.invoke(&mut input).await.unwrap();

    println!("{}", output);
}
