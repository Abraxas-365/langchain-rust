//! Offline tests for the semantic router (utils, MemoryIndex, RouteLayer) using a fake embedder.
use crate::{
    semantic_router::{
        utils::{combine_embeddings, cosine_similarity, sum_vectors},
        AggregationMethod, Index, IndexError, MemoryIndex, RouteLayerBuilder,
        RouteLayerBuilderError, Router,
    },
    test_utils::{FakeEmbedder, FakeLLM},
};

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
}

#[test]
fn vector_utils() {
    assert!(approx(cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]), 1.0));
    assert!(approx(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0));
    assert!(approx(cosine_similarity(&[1.0, 1.0], &[-1.0, -1.0]), -1.0));
    assert_eq!(
        combine_embeddings(&[vec![1.0, 3.0], vec![3.0, 5.0]]),
        vec![2.0, 4.0]
    );
    assert_eq!(
        sum_vectors(&[vec![1.0, 2.0], vec![3.0, 4.0]]),
        vec![4.0, 6.0]
    );
}

#[test]
fn aggregation_methods() {
    let v = [0.2, 0.8, 0.5];
    assert!(approx(AggregationMethod::Sum.aggregate(&v), 1.5));
    assert!(approx(AggregationMethod::Mean.aggregate(&v), 0.5));
    assert!(approx(AggregationMethod::Max.aggregate(&v), 0.8));
    assert!(approx(AggregationMethod::Max.aggregate(&[]), 0.0));
}

#[test]
fn router_equality_is_by_name() {
    let a = Router::new("a", &["x"]);
    let b = Router::new("a", &["y"]).with_similarity(0.5);
    assert_eq!(a, b);
    let mut set = std::collections::HashSet::new();
    set.insert(a);
    set.insert(b);
    assert_eq!(set.len(), 1);
}

#[tokio::test]
async fn memory_index_crud_and_query() {
    let mut index = MemoryIndex::new();
    let missing = index.add(&[Router::new("no_emb", &["x"])]).await;
    assert!(matches!(missing, Err(IndexError::MissingEmbedding(n)) if n == "no_emb"));

    index
        .add(&[
            Router::new("east", &["e"]).with_embedding(vec![vec![1.0, 0.0]]),
            Router::new("north", &["n", "n2"]).with_embedding(vec![vec![0.0, 1.0], vec![0.1, 1.0]]),
        ])
        .await
        .unwrap();
    assert_eq!(index.get_routers().await.unwrap().len(), 2);

    let top = index.query(&[0.0, 1.0], 2).await.unwrap();
    assert_eq!(top.len(), 2);
    assert!(top.iter().all(|(n, _)| n == "north"));
    assert!(top[0].1 >= top[1].1, "sorted descending");

    let all = index.query(&[1.0, 0.0], 10).await.unwrap();
    assert_eq!(all.len(), 3);
    assert_eq!(all[0].0, "east");

    assert_eq!(index.get_router("east").await.unwrap().name, "east");
    index.delete("east").await.unwrap();
    assert!(matches!(
        index.get_router("east").await,
        Err(IndexError::RouterNotFound(_))
    ));
    // deleting unknown is not an error
    index.delete("nope").await.unwrap();

    index.delete_index().await.unwrap();
    assert!(index.get_routers().await.unwrap().is_empty());
}

fn embedder() -> FakeEmbedder {
    FakeEmbedder::new(
        3,
        &[
            ("hello", vec![1.0, 0.0, 0.0]),
            ("hi there", vec![0.95, 0.05, 0.0]),
            ("weather in lima", vec![0.0, 1.0, 0.0]),
            ("is it raining", vec![0.05, 0.95, 0.0]),
            ("hey!", vec![0.9, 0.1, 0.0]),
            ("rain tomorrow in lima?", vec![0.0, 0.9, 0.1]),
        ],
    )
}

#[tokio::test]
async fn route_layer_picks_best_route() {
    let layer = RouteLayerBuilder::new()
        .embedder(embedder())
        .llm(FakeLLM::default())
        .index(MemoryIndex::new())
        .threshold(0.8)
        .add_route(Router::new("greeting", &["hello", "hi there"]))
        .add_route(Router::new(
            "weather",
            &["weather in lima", "is it raining"],
        ))
        .build()
        .await
        .unwrap();

    let choice = layer.call("hey!").await.unwrap().unwrap();
    assert_eq!(choice.route, "greeting");
    assert!(choice.similarity_score > 0.9);
    assert!(choice.tool_input.is_none());

    // Unknown text embeds to an orthogonal vector -> below threshold.
    assert!(layer.call("something else").await.unwrap().is_none());
    assert_eq!(layer.get_routers().await.unwrap().len(), 2);
}

#[tokio::test]
async fn route_layer_generates_tool_input() {
    let llm = FakeLLM::new([r#"{"city": "Lima"}"#, "plain text input"]);
    let layer = RouteLayerBuilder::new()
        .embedder(embedder())
        .llm(llm.clone())
        .index(MemoryIndex::new())
        .add_route(
            Router::new("weather", &["weather in lima", "is it raining"])
                .with_tool_description("Weather API, input is a JSON with city"),
        )
        .build()
        .await
        .unwrap();

    let choice = layer.call("rain tomorrow in lima?").await.unwrap().unwrap();
    assert_eq!(choice.route, "weather");
    assert_eq!(choice.tool_input, Some(serde_json::json!({"city": "Lima"})));
    let prompt = &llm.calls()[0][0].content;
    assert!(prompt.contains("Weather API"));
    assert!(prompt.contains("rain tomorrow in lima?"));

    // non-JSON LLM output is returned as a string value
    let choice = layer.call("rain tomorrow in lima?").await.unwrap().unwrap();
    assert_eq!(
        choice.tool_input,
        Some(serde_json::json!("plain text input"))
    );
}

#[tokio::test]
async fn route_layer_add_and_delete_routes() {
    let mut layer = RouteLayerBuilder::new()
        .embedder(embedder())
        .llm(FakeLLM::default())
        .index(MemoryIndex::new())
        .threshold(0.8)
        .build()
        .await
        .unwrap();
    assert!(layer.call("hey!").await.unwrap().is_none());

    let mut routes = [Router::new("greeting", &["hello"])];
    layer.add_routes(&mut routes).await.unwrap();
    assert!(routes[0].embedding.is_some());
    assert_eq!(layer.call("hey!").await.unwrap().unwrap().route, "greeting");

    layer.delete_route("greeting").await.unwrap();
    assert!(layer.call("hey!").await.unwrap().is_none());
}

#[tokio::test]
async fn route_layer_builder_validation() {
    let err = RouteLayerBuilder::new()
        .llm(FakeLLM::default())
        .index(MemoryIndex::new())
        .build()
        .await
        .err()
        .unwrap();
    assert!(matches!(err, RouteLayerBuilderError::MissingEmbedder));

    let err = RouteLayerBuilder::new()
        .embedder(embedder())
        .index(MemoryIndex::new())
        .build()
        .await
        .err()
        .unwrap();
    assert!(matches!(err, RouteLayerBuilderError::MissingLLM));

    let err = RouteLayerBuilder::new()
        .embedder(embedder())
        .llm(FakeLLM::default())
        .build()
        .await
        .err()
        .unwrap();
    assert!(matches!(err, RouteLayerBuilderError::MissingIndex));
}
