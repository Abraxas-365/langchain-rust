use std::{sync::Arc, time::Duration};

use mongodb::{
    bson::{doc, Bson, Document as BsonDocument},
    options::ClientOptions,
    Client, Collection,
};
use serde_json::json;
use tokio::sync::Mutex;

use super::*;
use crate::{
    chain::{builder::ConversationalChainBuilder, Chain},
    schemas::{FunctionCallResponse, ImageContent, MessageType},
    test_utils::FakeLLM,
};

/// A collection handle that never connects.
fn offline_collection() -> Collection<BsonDocument> {
    Client::with_options(ClientOptions::builder().build())
        .unwrap()
        .database("app")
        .collection("chat")
}

fn tool_call_message() -> Message {
    let call: FunctionCallResponse = serde_json::from_value(json!({
        "id": "call_1",
        "type": "function",
        "function": { "name": "search", "arguments": "{\"q\":\"rust\"}" }
    }))
    .unwrap();
    Message::new_ai_message("").with_tool_calls(json!([call]))
}

#[tokio::test]
async fn stored_documents_embed_the_message() {
    let memory = MongoMessageHistory::new(offline_collection(), "s1");
    assert_eq!(memory.session_id(), "s1");
    assert_eq!(memory.collection().name(), "chat");
    assert_eq!(memory.session_filter(), doc! { "session_id": "s1" });

    let stored = memory.to_stored(&Message::new_human_message("hi")).unwrap();
    assert_eq!(
        stored.keys().collect::<Vec<_>>(),
        ["session_id", "message", "created_at"]
    );
    assert_eq!(stored.get_str("session_id").unwrap(), "s1");
    assert!(matches!(stored.get("created_at"), Some(Bson::DateTime(_))));
    let age = mongodb::bson::DateTime::now().timestamp_millis()
        - stored
            .get_datetime("created_at")
            .unwrap()
            .timestamp_millis();
    assert!((0..5_000).contains(&age), "{age}");
    // A real embedded document (queryable), not a JSON string.
    let message = stored.get_document("message").unwrap();
    assert_eq!(message.get_str("content").unwrap(), "hi");
    assert_eq!(message.get_str("message_type").unwrap(), "human");
}

#[tokio::test]
async fn every_message_kind_round_trips() {
    let memory = MongoMessageHistory::new(offline_collection(), "s1");
    let mut with_image = Message::new_human_message_with_images(vec![ImageContent {
        image_url: "data:image/png;base64,AAA".into(),
        detail: Some("low".into()),
    }]);
    with_image.content = "what's this?".into();
    let messages = [
        Message::new_system_message("be nice"),
        Message::new_human_message("hi"),
        tool_call_message(),
        Message::new_tool_message("42", "call_1"),
        with_image,
    ];
    for message in messages {
        let back = MongoMessageHistory::from_stored(&memory.to_stored(&message).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_value(&back).unwrap(),
            serde_json::to_value(&message).unwrap()
        );
    }
}

#[test]
fn malformed_documents_are_errors() {
    let err = MongoMessageHistory::from_stored(&doc! { "session_id": "s1" }).unwrap_err();
    assert!(err.to_string().contains("no `message` document"), "{err}");
    let err = MongoMessageHistory::from_stored(&doc! { "message": { "content": 3 } }).unwrap_err();
    assert!(
        err.to_string().contains("invalid stored chat message"),
        "{err}"
    );
    let err = MongoMessageHistory::from_stored(&doc! { "message": "hi" }).unwrap_err();
    assert!(err.to_string().contains("no `message` document"), "{err}");
}

#[tokio::test]
async fn index_and_collection_options() {
    let model = MongoMessageHistory::index_model();
    assert_eq!(model.keys, doc! { "session_id": 1, "_id": 1 });
    assert_eq!(
        model.keys.keys().collect::<Vec<_>>(),
        ["session_id", "_id"],
        "compound index key order matters"
    );
    assert_eq!(
        model.options.unwrap().name.as_deref(),
        Some("session_id_1__id_1")
    );

    let memory = MongoMessageHistory::new(offline_collection(), "s1").with_collection("db2", "c2");
    assert_eq!(memory.collection().namespace().to_string(), "db2.c2");
    assert!(memory.create_index);
    assert!(!memory.without_index().create_index);

    let find = MongoMessageHistory::find_options();
    assert_eq!(find.sort, Some(doc! { "_id": 1 }), "insertion order");
    assert_eq!(find.projection, Some(doc! { "message": 1 }));
}

#[tokio::test]
async fn from_uri_uses_the_default_collection() {
    // Connecting is lazy, so this doesn't need a server.
    let memory = MongoMessageHistory::from_uri("mongodb://127.0.0.1:9", "s1")
        .await
        .unwrap();
    assert_eq!(
        memory.collection().namespace().to_string(),
        "chat_history.message_store"
    );
}

#[tokio::test]
async fn unreachable_servers_are_storage_errors() {
    // Nothing listens on port 9; fail fast instead of the default 30s.
    let client = Client::with_uri_str("mongodb://127.0.0.1:9/?serverSelectionTimeoutMS=200")
        .await
        .unwrap();
    let mut memory = MongoMessageHistory::new(client.database("db").collection("c"), "s1");
    let err = memory.messages().await.unwrap_err();
    assert!(matches!(err, MemoryError::Storage(_)), "{err:?}");
    let err = memory.add_user_message(&"hi").await.unwrap_err();
    assert!(matches!(err, MemoryError::Storage(_)), "{err:?}");
    let err = memory.clear().await.unwrap_err();
    assert!(matches!(err, MemoryError::Storage(_)), "{err:?}");
}

fn live_uri() -> String {
    std::env::var("MONGODB_HISTORY_TEST_URI")
        .or_else(|_| std::env::var("MONGODB_VECTOR_TEST_URI"))
        .expect("set MONGODB_HISTORY_TEST_URI")
}

/// Needs any MongoDB, e.g. `docker run -d -p 27017:27017 mongo:8` with
/// `MONGODB_HISTORY_TEST_URI=mongodb://localhost:27017/?directConnection=true`, then
/// `cargo test --features mongodb -- --ignored mongodb_history_live`.
#[tokio::test]
#[ignore]
async fn mongodb_history_live_round_trip() {
    let client = Client::with_uri_str(live_uri()).await.unwrap();
    let database = format!(
        "lcr_history_{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let collection = client
        .database(&database)
        .collection::<BsonDocument>("chat");

    let mut a = MongoMessageHistory::new(collection.clone(), "a");
    let mut b = MongoMessageHistory::new(collection.clone(), "b");
    assert!(a.messages().await.unwrap().is_empty());

    // Enough messages in one session to catch any ordering that isn't insertion order.
    for i in 0..20 {
        a.add_user_message(&format!("q{i}")).await.unwrap();
        a.add_ai_message(&format!("a{i}")).await.unwrap();
    }
    a.add_message(tool_call_message()).await.unwrap();
    a.add_message(Message::new_tool_message("42", "call_1"))
        .await
        .unwrap();
    b.add_user_message(&"hi from b").await.unwrap();

    // Order comes from `_id`, not from the natural (insertion) order: insert out of order, in a collection without the session index (an index scan would
    // return `_id` order by itself).
    let unindexed = client
        .database(&database)
        .collection::<BsonDocument>("unindexed");
    let raw_memory = MongoMessageHistory::new(unindexed.clone(), "raw").without_index();
    for (id, content) in [(2u8, "second"), (1, "first")] {
        let mut stored = raw_memory
            .to_stored(&Message::new_human_message(content))
            .unwrap();
        let mut bytes = [0u8; 12];
        bytes[11] = id;
        stored.insert("_id", mongodb::bson::oid::ObjectId::from_bytes(bytes));
        unindexed.insert_one(stored).await.unwrap();
    }
    let contents: Vec<_> = raw_memory
        .messages()
        .await
        .unwrap()
        .into_iter()
        .map(|m| m.content)
        .collect();
    assert_eq!(contents, ["first", "second"]);

    // A fresh instance sees what was persisted, in order.
    let reopened = MongoMessageHistory::from_uri(&live_uri(), "a")
        .await
        .unwrap()
        .with_collection(&database, "chat");
    let messages = reopened.messages().await.unwrap();
    assert_eq!(messages.len(), 42);
    let expected: Vec<String> = (0..20)
        .flat_map(|i| [format!("q{i}"), format!("a{i}")])
        .collect();
    let contents: Vec<_> = messages[..40].iter().map(|m| m.content.clone()).collect();
    assert_eq!(contents, expected);
    assert_eq!(messages[1].message_type, MessageType::AIMessage);
    assert_eq!(messages[40].tool_calls, tool_call_message().tool_calls);
    assert_eq!(messages[41].message_type, MessageType::ToolMessage);
    assert_eq!(messages[41].id.as_deref(), Some("call_1"));
    assert_eq!(b.messages().await.unwrap().len(), 1);

    // The session index exists, and documents have the documented shape.
    let names = collection.list_index_names().await.unwrap();
    assert!(
        names.contains(&"session_id_1__id_1".to_string()),
        "{names:?}"
    );
    let raw = collection
        .find_one(doc! { "session_id": "b" })
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        raw.get_document("message")
            .unwrap()
            .get_str("content")
            .unwrap(),
        "hi from b"
    );
    assert!(raw.get_datetime("created_at").is_ok());

    // clear only touches its own session.
    a.clear().await.unwrap();
    assert!(a.messages().await.unwrap().is_empty());
    assert_eq!(b.messages().await.unwrap().len(), 1);

    // Drives a conversational chain across "restarts".
    let llm = FakeLLM::new(["Hi Ana", "You're Ana"]);
    for (input, answer) in [("I'm Ana", "Hi Ana"), ("Who am I?", "You're Ana")] {
        let memory: Arc<Mutex<dyn BaseMemory>> =
            MongoMessageHistory::new(collection.clone(), "chain").into();
        let chain = ConversationalChainBuilder::new()
            .llm(llm.clone())
            .memory(memory)
            .build()
            .unwrap();
        let got = chain
            .invoke(crate::prompt_args! { "input" => input })
            .await
            .unwrap();
        assert_eq!(got, answer);
    }
    let second_prompt = &llm.calls()[1];
    let texts: Vec<_> = second_prompt.iter().map(|m| m.content.as_str()).collect();
    assert!(
        texts.iter().any(|t| t.contains("I'm Ana")) && texts.iter().any(|t| t.contains("Hi Ana")),
        "history from the first instance reached the LLM: {texts:?}"
    );

    // without_index on a fresh collection creates nothing.
    let bare = client
        .database(&database)
        .collection::<BsonDocument>("bare");
    let mut memory = MongoMessageHistory::new(bare.clone(), "x").without_index();
    memory.add_user_message(&"hi").await.unwrap();
    assert_eq!(bare.list_index_names().await.unwrap(), ["_id_"]);

    // Reads never create the index (read-only users); the first write does, also after
    // switching collections on an instance that already created one elsewhere.
    let mut moved = MongoMessageHistory::new(collection.clone(), "m");
    moved.add_user_message(&"hi").await.unwrap();
    let mut moved = moved.with_collection(&database, "moved");
    assert!(moved.messages().await.unwrap().is_empty());
    let other = client
        .database(&database)
        .collection::<BsonDocument>("moved");
    assert!(other
        .list_index_names()
        .await
        .unwrap_or_default()
        .is_empty());
    moved.add_user_message(&"hi").await.unwrap();
    assert!(other
        .list_index_names()
        .await
        .unwrap()
        .contains(&"session_id_1__id_1".to_string()));

    tokio::time::timeout(Duration::from_secs(30), client.database(&database).drop())
        .await
        .unwrap()
        .unwrap();
}
