// Few-shot prompting: show the model a few examples before the input. A fixed list works,
// or an example selector picks the examples for each input, here the most similar ones by
// embedding (Ollama's nomic-embed-text, `ollama pull nomic-embed-text`).
//
// OpenAI:            OPENAI_API_KEY=... cargo run --example few_shot --features ollama
// Any local model:   FEW_SHOT_OLLAMA_MODEL=qwen3:14b cargo run --example few_shot --features ollama
//                    (talks to Ollama's OpenAI-compatible endpoint on localhost:11434)

#[cfg(feature = "ollama")]
use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    embedding::OllamaEmbedder,
    llm::{openai::OpenAI, OpenAIConfig},
    prompt::{FewShotPromptTemplate, SemanticSimilarityExampleSelector},
    prompt_args, template_fstring,
};

#[cfg(feature = "ollama")]
#[tokio::main]
async fn main() {
    let llm = match std::env::var("FEW_SHOT_OLLAMA_MODEL") {
        Ok(model) => OpenAI::default()
            .with_config(
                OpenAIConfig::default()
                    .with_api_base("http://localhost:11434/v1")
                    .with_api_key("ollama"),
            )
            .with_model(model),
        Err(_) => OpenAI::default(),
    };

    let examples = vec![
        prompt_args! {"input" => "apple", "output" => "FRUIT::apple::red"},
        prompt_args! {"input" => "banana", "output" => "FRUIT::banana::yellow"},
        prompt_args! {"input" => "carrot", "output" => "VEGETABLE::carrot::orange"},
        prompt_args! {"input" => "spinach", "output" => "VEGETABLE::spinach::green"},
        prompt_args! {"input" => "truck", "output" => "VEHICLE::truck::any"},
        prompt_args! {"input" => "bicycle", "output" => "VEHICLE::bicycle::any"},
    ];
    // Pick the 2 examples closest to the input, comparing only the "input" field.
    let selector =
        SemanticSimilarityExampleSelector::from_examples(examples, OllamaEmbedder::default(), 2)
            .await
            .unwrap()
            .with_input_keys(["input"]);

    let prompt = FewShotPromptTemplate::with_selector(
        template_fstring!("Input: {input}\nOutput: {output}", "input", "output"),
        selector,
    )
    .with_prefix(template_fstring!(
        "Classify the input exactly like the examples. Reply with the output line only.",
    ))
    .with_suffix(template_fstring!("Input: {input}\nOutput:", "input"));

    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(llm)
        .build()
        .unwrap();

    for input in ["motorcycle", "pear", "broccoli"] {
        let answer = chain.invoke(prompt_args! {"input" => input}).await.unwrap();
        println!("{input} -> {}", answer.trim());
    }
}

#[cfg(not(feature = "ollama"))]
fn main() {
    println!("This example uses Ollama embeddings: cargo run --example few_shot --features ollama");
}
