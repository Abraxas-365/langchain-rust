use langchain_rust::{
    chain::{chain_trait::Chain, llm_chain::LLMChainBuilder},
    language_models::options::CallOptions,
    llm::{Qwen, QwenModel},
    prompt::{PromptTemplate, TemplateFormat},
    prompt_args,
};
use std::env;

#[tokio::main]
async fn main() {
    // Get API key from environment variable
    let api_key = env::var("QWEN_API_KEY").expect("QWEN_API_KEY environment variable must be set");

    // Setup the Qwen client with desired model and parameters
    let qwen = Qwen::new()
        .with_api_key(api_key)
        .with_model(QwenModel::QwenTurbo.to_string())
        .with_options(
            CallOptions::default()
                .with_max_tokens(800)
                .with_temperature(0.8),
        );

    // Create a prompt template
    let template = r#"
    You are a helpful assistant that provides detailed information. 
    
    User question: {question}
    
    Please provide a comprehensive answer:
    "#;

    let prompt = PromptTemplate::new(
        template.to_owned(),
        vec!["question".to_owned()],
        TemplateFormat::FString,
    );

    // Create an LLMChain using the builder pattern
    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(qwen)
        .build()
        .unwrap();

    // Execute the chain with a question
    let inputs = prompt_args! {
        "question" => "Explain the importance of quantum computing and its potential applications."
    };

    let result = chain.call(inputs).await.unwrap();

    println!(
        "Question: Explain the importance of quantum computing and its potential applications."
    );
    println!("\nQwen's response:");
    println!("{}", result.generation);
}
