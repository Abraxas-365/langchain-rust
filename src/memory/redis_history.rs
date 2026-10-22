//! Persistent chat history in Redis (feature `redis`).
//!
//! Each session is a Redis list at `{key_prefix}{session_id}` holding one JSON message per
//! entry, so tool calls, tool call ids and images survive a restart. An optional TTL is
//! refreshed on every write, so idle conversations expire.

use std::time::Duration;

use async_trait::async_trait;
use redis::{aio::ConnectionLike, aio::ConnectionManager, AsyncCommands};

use crate::schemas::{
    memory::{BaseMemory, MemoryError},
    messages::Message,
};

pub const DEFAULT_REDIS_KEY_PREFIX: &str = "langchain:message_history:";

/// Chat history stored in a Redis list.
///
/// ```rust,ignore
/// let memory = RedisMessageHistory::from_url("redis://127.0.0.1/", "user-42")
///     .await?
///     .with_ttl(Duration::from_secs(24 * 3600));
/// let chain = ConversationalChainBuilder::new().llm(llm).memory(memory.into()).build()?;
/// ```
///
/// Any async Redis connection works (`ConnectionManager`, `MultiplexedConnection`, a cluster
/// connection...); `from_url` uses a `ConnectionManager`, which reconnects on its own.
pub struct RedisMessageHistory<C = ConnectionManager> {
    connection: C,
    session_id: String,
    key: String,
    ttl: Option<Duration>,
}

impl RedisMessageHistory<ConnectionManager> {
    /// Connects to `url` (e.g. `redis://127.0.0.1/` or `rediss://...` with TLS support
    /// enabled in your `redis` features).
    pub async fn from_url(url: &str, session_id: impl Into<String>) -> Result<Self, MemoryError> {
        let client = redis::Client::open(url)?;
        let connection = ConnectionManager::new(client).await?;
        Ok(Self::new(connection, session_id))
    }
}

impl<C> RedisMessageHistory<C>
where
    C: ConnectionLike + Clone + Send + Sync,
{
    pub fn new(connection: C, session_id: impl Into<String>) -> Self {
        let session_id = session_id.into();
        Self {
            connection,
            key: format!("{DEFAULT_REDIS_KEY_PREFIX}{session_id}"),
            session_id,
            ttl: None,
        }
    }

    /// Key prefix; the list lives at `{prefix}{session_id}`.
    /// Default: [`DEFAULT_REDIS_KEY_PREFIX`].
    pub fn with_key_prefix(mut self, prefix: impl AsRef<str>) -> Self {
        self.key = format!("{}{}", prefix.as_ref(), self.session_id);
        self
    }

    /// Expire the history after `ttl` without writes (whole seconds, at least 1).
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The Redis key holding this session's messages.
    pub fn key(&self) -> &str {
        &self.key
    }
}

#[async_trait]
impl<C> BaseMemory for RedisMessageHistory<C>
where
    C: ConnectionLike + Clone + Send + Sync,
{
    async fn messages(&self) -> Result<Vec<Message>, MemoryError> {
        let mut connection = self.connection.clone();
        let rows: Vec<String> = connection.lrange(&self.key, 0, -1).await?;
        rows.iter()
            .map(|raw| serde_json::from_str(raw).map_err(MemoryError::from))
            .collect()
    }

    async fn add_message(&mut self, message: Message) -> Result<(), MemoryError> {
        let raw = serde_json::to_string(&message)?;
        match self.ttl {
            None => {
                let _: () = self.connection.rpush(&self.key, raw).await?;
            }
            Some(ttl) => {
                let seconds = ttl.as_secs().max(1) as i64;
                let _: () = redis::pipe()
                    .atomic()
                    .rpush(&self.key, raw)
                    .ignore()
                    .expire(&self.key, seconds)
                    .ignore()
                    .query_async(&mut self.connection)
                    .await?;
            }
        }
        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        let _: () = self.connection.del(&self.key).await?;
        Ok(())
    }
}

impl<C> From<RedisMessageHistory<C>> for std::sync::Arc<tokio::sync::Mutex<dyn BaseMemory>>
where
    C: ConnectionLike + Clone + Send + Sync + 'static,
{
    fn from(memory: RedisMessageHistory<C>) -> Self {
        std::sync::Arc::new(tokio::sync::Mutex::new(memory))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use redis::{cmd, pipe, ErrorKind, RedisError};
    use redis_test::{MockCmd, MockRedisConnection};
    use serde_json::json;
    use tokio::sync::Mutex;

    use super::*;
    use crate::{
        chain::{builder::ConversationalChainBuilder, Chain},
        schemas::{FunctionCallResponse, MessageType},
        test_utils::FakeLLM,
    };

    const KEY: &str = "langchain:message_history:s1";

    fn json_of(message: &Message) -> String {
        serde_json::to_string(message).unwrap()
    }

    fn list<const N: usize>(items: [String; N]) -> redis::Value {
        redis::Value::Array(
            items
                .into_iter()
                .map(|s| redis::Value::BulkString(s.into_bytes()))
                .collect(),
        )
    }

    #[tokio::test]
    async fn add_appends_json_and_messages_reads_the_list_in_order() {
        let human = Message::new_human_message("hi");
        let ai = Message::new_ai_message("hello");
        let conn = MockRedisConnection::new([
            MockCmd::new(cmd("RPUSH").arg(KEY).arg(json_of(&human)), Ok(1)),
            MockCmd::new(cmd("RPUSH").arg(KEY).arg(json_of(&ai)), Ok(2)),
            MockCmd::new(
                cmd("LRANGE").arg(KEY).arg(0).arg(-1),
                Ok(list([json_of(&human), json_of(&ai)])),
            ),
        ])
        .assert_all_commands_consumed();

        let mut memory = RedisMessageHistory::new(conn, "s1");
        assert_eq!(memory.key(), KEY);
        assert_eq!(memory.session_id(), "s1");
        memory.add_user_message(&"hi").await.unwrap();
        memory.add_ai_message(&"hello").await.unwrap();

        let messages = memory.messages().await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].message_type, MessageType::HumanMessage);
        assert_eq!(messages[0].content, "hi");
        assert_eq!(messages[1].message_type, MessageType::AIMessage);
        assert_eq!(messages[1].content, "hello");
    }

    #[tokio::test]
    async fn ttl_pushes_and_expires_atomically() {
        let message = Message::new_human_message("hi");
        let conn = MockRedisConnection::new([MockCmd::with_values(
            pipe()
                .atomic()
                .cmd("RPUSH")
                .arg("chat:s1")
                .arg(json_of(&message))
                .ignore()
                .cmd("EXPIRE")
                .arg("chat:s1")
                .arg(90)
                .ignore(),
            Ok(vec![
                redis::Value::Okay,
                redis::Value::Okay,
                redis::Value::Okay,
                redis::Value::Array(vec![]),
            ]),
        )])
        .assert_all_commands_consumed();

        let mut memory = RedisMessageHistory::new(conn, "s1")
            .with_key_prefix("chat:")
            .with_ttl(Duration::from_secs(90));
        assert_eq!(memory.key(), "chat:s1");
        memory.add_message(message).await.unwrap();
    }

    #[tokio::test]
    async fn clear_deletes_the_key() {
        let conn = MockRedisConnection::new([MockCmd::new(cmd("DEL").arg(KEY), Ok(1))])
            .assert_all_commands_consumed();
        RedisMessageHistory::new(conn, "s1").clear().await.unwrap();
    }

    #[tokio::test]
    async fn redis_and_decoding_errors_surface_as_memory_errors() {
        let conn = MockRedisConnection::new([
            MockCmd::new(
                cmd("LRANGE").arg(KEY).arg(0).arg(-1),
                Err::<String, _>(RedisError::from((ErrorKind::Io, "connection refused"))),
            ),
            MockCmd::new(
                cmd("LRANGE").arg(KEY).arg(0).arg(-1),
                Ok(list(["not json".to_string()])),
            ),
        ]);
        let memory = RedisMessageHistory::new(conn, "s1");

        let err = memory.messages().await.unwrap_err();
        assert!(matches!(err, MemoryError::Storage(_)), "{err:?}");
        assert!(err.to_string().contains("connection refused"), "{err}");

        let err = memory.messages().await.unwrap_err();
        assert!(matches!(err, MemoryError::Serialization(_)), "{err:?}");
    }

    #[tokio::test]
    async fn tool_calls_round_trip_through_the_stored_json() {
        let call: FunctionCallResponse = serde_json::from_value(json!({
            "id": "call_1",
            "type": "function",
            "function": { "name": "search", "arguments": "{\"q\":\"rust\"}" }
        }))
        .unwrap();
        let ai = Message::new_ai_message("").with_tool_calls(json!([call]));
        let tool = Message::new_tool_message("42", "call_1");
        let conn = MockRedisConnection::new([MockCmd::new(
            cmd("LRANGE").arg(KEY).arg(0).arg(-1),
            Ok(list([json_of(&ai), json_of(&tool)])),
        )]);

        let messages = RedisMessageHistory::new(conn, "s1")
            .messages()
            .await
            .unwrap();
        assert_eq!(messages[0].tool_calls, ai.tool_calls);
        assert_eq!(messages[1].message_type, MessageType::ToolMessage);
        assert_eq!(messages[1].id.as_deref(), Some("call_1"));
    }

    #[tokio::test]
    async fn redis_history_drives_a_conversational_chain() {
        let conn = MockRedisConnection::new([
            // Turn 1: load (empty), then save both sides.
            MockCmd::new(cmd("LRANGE").arg(KEY).arg(0).arg(-1), Ok(list([]))),
            MockCmd::new(
                cmd("RPUSH")
                    .arg(KEY)
                    .arg(json_of(&Message::new_human_message("I'm Ana"))),
                Ok(1),
            ),
            MockCmd::new(
                cmd("RPUSH")
                    .arg(KEY)
                    .arg(json_of(&Message::new_ai_message("Hi Ana"))),
                Ok(2),
            ),
        ])
        .assert_all_commands_consumed();
        let memory: Arc<Mutex<dyn BaseMemory>> = RedisMessageHistory::new(conn, "s1").into();
        let chain = ConversationalChainBuilder::new()
            .llm(FakeLLM::new(["Hi Ana"]))
            .memory(memory)
            .build()
            .unwrap();
        let answer = chain
            .invoke(crate::prompt_args! { "input" => "I'm Ana" })
            .await
            .unwrap();
        assert_eq!(answer, "Hi Ana");
    }

    /// Needs a local Redis: `redis-server --port 6379` (or `REDIS_TEST_URL`), then
    /// `cargo test --features redis -- --ignored redis_live`.
    #[tokio::test]
    #[ignore]
    async fn redis_live_round_trip_sessions_and_ttl() {
        let url = std::env::var("REDIS_TEST_URL").unwrap_or("redis://127.0.0.1:6379/".into());
        let prefix = format!("langchain-rust-test:{}:", uuid_like());
        let mut a = RedisMessageHistory::from_url(&url, "a")
            .await
            .unwrap()
            .with_key_prefix(&prefix)
            .with_ttl(Duration::from_secs(60));
        let mut b = RedisMessageHistory::from_url(&url, "b")
            .await
            .unwrap()
            .with_key_prefix(&prefix);

        a.add_user_message(&"hi from a").await.unwrap();
        a.add_ai_message(&"hello a").await.unwrap();
        b.add_user_message(&"hi from b").await.unwrap();

        // A fresh instance sees what was persisted.
        let reopened = RedisMessageHistory::from_url(&url, "a")
            .await
            .unwrap()
            .with_key_prefix(&prefix);
        let contents: Vec<String> = reopened
            .messages()
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.content)
            .collect();
        assert_eq!(contents, ["hi from a", "hello a"]);
        assert_eq!(b.messages().await.unwrap().len(), 1);

        let client = redis::Client::open(url.as_str()).unwrap();
        let mut conn = client.get_multiplexed_async_connection().await.unwrap();
        let ttl_a: i64 = conn.ttl(a.key()).await.unwrap();
        let ttl_b: i64 = conn.ttl(b.key()).await.unwrap();
        assert!((1..=60).contains(&ttl_a), "{ttl_a}");
        assert_eq!(ttl_b, -1, "no TTL configured for b");

        a.clear().await.unwrap();
        b.clear().await.unwrap();
        assert!(a.messages().await.unwrap().is_empty());
        let exists: bool = conn.exists(b.key()).await.unwrap();
        assert!(!exists);
    }

    fn uuid_like() -> String {
        format!(
            "{:x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )
    }
}
