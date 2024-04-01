use langchain_rust::{
    embedding::openai::OpenAiEmbedder,
    semantic_router::{AggregationMethod, RouteLayerBuilder, Router},
};

#[tokio::main]
async fn main() {
    let captial_route = Router::new(
        "captial",
        &[
            "Capital of France is Paris.",
            "What is the captial of France?",
        ],
    );
    let weather_route = Router::new(
        "temperature",
        &[
            "What is the temperature?",
            "Is it raining?",
            "Is it cloudy?",
        ],
    );
    let router_layer = RouteLayerBuilder::default()
        .embedder(OpenAiEmbedder::default())
        .add_route(captial_route)
        .add_route(weather_route)
        .aggregation_method(AggregationMethod::Sum)
        .threshold(0.82)
        .build()
        .await
        .unwrap();

    let routes = router_layer
        .call("What is the temperature in Peru?")
        .await
        .unwrap();

    println!("{:?}", routes);
}
