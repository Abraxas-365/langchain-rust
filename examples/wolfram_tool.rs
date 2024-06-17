use langchain_rust::tools::{Tool, Wolfram};

#[tokio::main]
async fn main() {
    let wolfram = Wolfram::default().with_excludes(&["Plot"]);
    let input = "Solve x^2 - 2x + 1 = 0";
    let result = wolfram.call(input).await;

    println!("{}", result.unwrap());
}
