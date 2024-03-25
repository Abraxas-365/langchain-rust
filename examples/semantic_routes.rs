use langchain_rust::{
    embedding::openai::OpenAiEmbedder,
    semantic_router::{RouteLayerBuilder, RouterBuilder},
};

#[tokio::main]
async fn main() {
    let politics_route = RouterBuilder::new("politics")
        .utterances(&[
            "isn't politics the best thing ever",
            "why don't you tell me about your political opinions",
            "don't you just love the president",
            "they're going to destroy this country!",
            "they will save the country!",
        ])
        .build()
        .unwrap();

    let chitchat_route = RouterBuilder::new("chitchat")
        .utterances(&[
            "how's the weather today?",
            "how are things going?",
            "lovely weather today",
            "the weather is horrendous",
            "let's go to the chippy",
        ])
        .build()
        .unwrap();

    let router_layer = RouteLayerBuilder::new()
        .add_route(politics_route)
        .add_route(chitchat_route)
        .embedder(OpenAiEmbedder::default())
        .threshold(0.7)
        .build()
        .await
        .unwrap();

    let most_similar = router_layer
        .route("isn't politics the best thing ever")
        .await
        .unwrap();

    if let Some(most_similar) = most_similar {
        println!("Most similar route: {}", most_similar.name);
    } else {
        println!("No route found");
    }
}
