use langchain_rust::{
    chain::{Chain, StuffDocument},
    llm::openai::OpenAI,
    schemas::Document,
};

#[tokio::main]
async fn main() {
    let llm = OpenAI::default();
    let chain = StuffDocument::load_stuff_qa(llm);
    let input = chain
        .qa_prompt_builder()
        //You could also get the documents form a retriver
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
        .question("How old is luis and whats his favorite text editor")
        .build();

    let ouput = chain.invoke(input).await.unwrap();

    println!("{}", ouput);
}
