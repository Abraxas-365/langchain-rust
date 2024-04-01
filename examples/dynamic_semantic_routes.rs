use langchain_rust::{
    embedding::openai::OpenAiEmbedder,
    semantic_router::{AggregationMethod, RouteLayerBuilder, Router},
    tools::{SerpApi, Tool},
};

#[tokio::main]
async fn main() {
    let tool = SerpApi::default();
    let captial_route = Router::new(
        "captial",
        &[
            "Capital of France is Paris.",
            "What is the captial of France?",
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
        .add_route(captial_route)
        .add_route(weather_route)
        .aggregation_method(AggregationMethod::Sum)
        .threshold(0.82)
        .build()
        .await
        .unwrap();

    let route = router_layer
        .call("What is the capital capital of USA")
        .await
        .unwrap();

    let route_choise = match route {
        Some(route) => route,
        None => panic!("No Similar Route"),
    };

    println!("{:?}", &route_choise);
    if route_choise.route == "captial" {
        let tool_ouput = tool.run(route_choise.tool_input.unwrap()).await.unwrap();
        println!("{:?}", tool_ouput);
    }
}
