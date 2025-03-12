use std::sync::Arc;

use langchain_rust::tools::{Tool, Wolfram};
use serde_json::Value;

#[tokio::main]
async fn main() {
    let wolfram: Arc<dyn Tool> = Wolfram::default().with_excludes(&["Plot"]).into();
    let input = "Solve x^2 - 2x + 1 = 0";
    let result = wolfram.call(Value::String(input.to_string())).await;

    println!("{}", result.unwrap());
}
