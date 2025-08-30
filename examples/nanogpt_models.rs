use langchain_rust::llm::nanogpt::*;
use dotenvy::dotenv;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() {
    dotenv().expect(".env file not found");
    
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).unwrap();

    let nano_config = OpenAIConfig::new()
        .with_api_base("https://nano-gpt.com/api/v1")
        .with_api_key(dotenvy::var("NANOGPT_KEY").expect("NANOGPT_KEY not set"));

    let client = NanoGPT::default().with_config(nano_config);

    // Basic model listing
    let basic_models = client.get_models(false)
        .await
        .unwrap_or_else(|e| panic!("Error fetching models: {:?}", e));

    println!("\nBasic Models:");
    for model in &basic_models.data {
        println!("- {} ({})", model.id, model.owned_by);
    }

    // Detailed model listing with pricing
    let detailed_models = client.get_models(true)
        .await
        .unwrap_or_else(|e| panic!("Error fetching detailed models: {:?}", e));

    println!("\nDetailed Models:");
    for model in detailed_models.data {
        println!("Model: {}", model.name.unwrap_or("Unnamed".into()));
        println!("ID: {}", model.id);
        if let Some(pricing) = model.pricing {
            println!("Pricing: ${}/M tokens (input), ${}/M tokens (output)", 
                pricing.prompt, pricing.completion);
        }
        println!("---");
    }
}
