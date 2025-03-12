use std::sync::Arc;

use langchain_rust::{
    embedding::openai::OpenAiEmbedder,
    semantic_router::{AggregationMethod, RouteLayerBuilder, Router},
    tools::{SerpApi, Tool},
};

#[tokio::main]
async fn main() {
    let tool: Arc<dyn Tool> = SerpApi::default().into();
    let capital_route = Router::new(
        "capital",
        &[
            "Capital of France is Paris.",
            "What is the capital of France?",
        ],
    )
    .with_tool_description(tool.description());
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
        .add_route(capital_route)
        .add_route(weather_route)
        .aggregation_method(AggregationMethod::Sum)
        .threshold(0.82)
        .build()
        .await
        .unwrap();

    let route = router_layer
        .call("What is the capital of USA")
        .await
        .unwrap();

    let route_choice = match route {
        Some(route) => route,
        None => panic!("No Similar Route"),
    };

    println!("{:?}", &route_choice);
    if route_choice.route == "capital" {
        let tool_output = tool.call(route_choice.tool_input.unwrap()).await.unwrap();
        println!("{:?}", tool_output);
    }
}
