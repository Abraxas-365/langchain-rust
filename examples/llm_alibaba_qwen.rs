use langchain_rust::language_models::llm::LLM;
use langchain_rust::llm::Qwen;
use langchain_rust::schemas::Message;

#[tokio::main]
async fn main() {
    // Initialize the Qwen client
    // Requires QWEN_API_KEY environment variable to be set
    let qwen = Qwen::new()
        .with_api_key("your_api_key")
        .with_model("qwen-turbo"); // Can use enum: QwenModel::QwenTurbo.to_string()

    // Generate a response
    let response = qwen
        .generate(&[Message::new_human_message("Introduce the Great Wall")])
        .await
        .unwrap();

    println!("Response: {}", response.generation);
}
