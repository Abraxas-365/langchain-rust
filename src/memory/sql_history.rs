//! Persistent chat histories backed by SQLite or Postgres (feature `sqlx`).
//!
//! Every message is stored as one JSON row keyed by `session_id`, so tool calls,
//! tool call ids and images survive a restart. The table is created on first use
//! unless [`SqliteMessageHistory::without_migration`] / [`PostgresMessageHistory::without_migration`]
//! is set.

use async_trait::async_trait;
use sqlx::{AssertSqlSafe, Pool, Postgres, Row, Sqlite};
use tokio::sync::OnceCell;

use crate::schemas::{
    memory::{BaseMemory, MemoryError},
    messages::Message,
};

pub const DEFAULT_MESSAGE_HISTORY_TABLE: &str = "langchain_message_history";

/// Table names are interpolated into SQL, so only allow plain identifiers
/// (optionally schema-qualified, e.g. `chat.history`).
fn validate_table_name(name: &str) -> Result<(), MemoryError> {
    let valid = !name.is_empty()
        && name.split('.').all(|part| {
            let mut chars = part.chars();
            matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        });
    if valid {
        Ok(())
    } else {
        Err(MemoryError::Other(format!("invalid table name: {name:?}")))
    }
}

fn decode(rows: Vec<String>) -> Result<Vec<Message>, MemoryError> {
    rows.iter()
        .map(|raw| serde_json::from_str(raw).map_err(MemoryError::from))
        .collect()
}

/// Chat history stored in SQLite.
///
/// ```rust,ignore
/// let pool = SqlitePool::connect("sqlite://chat.db?mode=rwc").await?;
/// let memory = SqliteMessageHistory::new(pool, "user-42");
/// let chain = ConversationalChainBuilder::new().llm(llm).memory(memory.into()).build()?;
/// ```
///
/// With `sqlite::memory:` use a pool with `max_connections(1)`, otherwise every
/// connection gets its own empty database.
pub struct SqliteMessageHistory {
    pool: Pool<Sqlite>,
    session_id: String,
    table: String,
    migrate: bool,
    ready: OnceCell<()>,
}

impl SqliteMessageHistory {
    pub fn new(pool: Pool<Sqlite>, session_id: impl Into<String>) -> Self {
        Self {
            pool,
            session_id: session_id.into(),
            table: DEFAULT_MESSAGE_HISTORY_TABLE.to_string(),
            migrate: true,
            ready: OnceCell::new(),
        }
    }

    /// Use a different table (letters, digits and `_` only).
    pub fn with_table_name(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    /// Do not create the table; it must already exist with the expected columns.
    pub fn without_migration(mut self) -> Self {
        self.migrate = false;
        self
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    async fn ensure_table(&self) -> Result<(), MemoryError> {
        self.ready
            .get_or_try_init(|| async {
                validate_table_name(&self.table)?;
                if self.migrate {
                    let table = &self.table;
                    sqlx::query(AssertSqlSafe(format!(
                        "CREATE TABLE IF NOT EXISTS {table} (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            session_id TEXT NOT NULL,
                            message TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                        )"
                    )))
                    .execute(&self.pool)
                    .await?;
                    let index = table.replace('.', "_");
                    sqlx::query(AssertSqlSafe(format!(
                        "CREATE INDEX IF NOT EXISTS {index}_session_idx ON {table} (session_id, id)"
                    )))
                    .execute(&self.pool)
                    .await?;
                }
                Ok::<(), MemoryError>(())
            })
            .await
            .map(|_| ())
    }
}

#[async_trait]
impl BaseMemory for SqliteMessageHistory {
    async fn messages(&self) -> Result<Vec<Message>, MemoryError> {
        self.ensure_table().await?;
        let rows = sqlx::query(AssertSqlSafe(format!(
            "SELECT message FROM {} WHERE session_id = ? ORDER BY id",
            self.table
        )))
        .bind(&self.session_id)
        .fetch_all(&self.pool)
        .await?;
        decode(
            rows.iter()
                .map(|r| r.try_get::<String, _>("message"))
                .collect::<Result<_, _>>()?,
        )
    }

    async fn add_message(&mut self, message: Message) -> Result<(), MemoryError> {
        self.ensure_table().await?;
        sqlx::query(AssertSqlSafe(format!(
            "INSERT INTO {} (session_id, message) VALUES (?, ?)",
            self.table
        )))
        .bind(&self.session_id)
        .bind(serde_json::to_string(&message)?)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        self.ensure_table().await?;
        sqlx::query(AssertSqlSafe(format!(
            "DELETE FROM {} WHERE session_id = ?",
            self.table
        )))
        .bind(&self.session_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

/// Chat history stored in Postgres.
///
/// ```rust,ignore
/// let pool = PgPool::connect("postgres://user:pw@localhost/db").await?;
/// let memory = PostgresMessageHistory::new(pool, "user-42");
/// let executor = AgentExecutor::from_agent(agent).with_memory(memory.into());
/// ```
pub struct PostgresMessageHistory {
    pool: Pool<Postgres>,
    session_id: String,
    table: String,
    migrate: bool,
    ready: OnceCell<()>,
}

impl PostgresMessageHistory {
    pub fn new(pool: Pool<Postgres>, session_id: impl Into<String>) -> Self {
        Self {
            pool,
            session_id: session_id.into(),
            table: DEFAULT_MESSAGE_HISTORY_TABLE.to_string(),
            migrate: true,
            ready: OnceCell::new(),
        }
    }

    /// Use a different table, optionally schema-qualified (`schema.table`).
    pub fn with_table_name(mut self, table: impl Into<String>) -> Self {
        self.table = table.into();
        self
    }

    /// Do not create the table; it must already exist with the expected columns.
    pub fn without_migration(mut self) -> Self {
        self.migrate = false;
        self
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    async fn ensure_table(&self) -> Result<(), MemoryError> {
        self.ready
            .get_or_try_init(|| async {
                validate_table_name(&self.table)?;
                if self.migrate {
                    let table = &self.table;
                    sqlx::query(AssertSqlSafe(format!(
                        "CREATE TABLE IF NOT EXISTS {table} (
                            id BIGSERIAL PRIMARY KEY,
                            session_id TEXT NOT NULL,
                            message TEXT NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                        )"
                    )))
                    .execute(&self.pool)
                    .await?;
                    let index = table.rsplit('.').next().unwrap_or(table);
                    sqlx::query(AssertSqlSafe(format!(
                        "CREATE INDEX IF NOT EXISTS {index}_session_idx ON {table} (session_id, id)"
                    )))
                    .execute(&self.pool)
                    .await?;
                }
                Ok::<(), MemoryError>(())
            })
            .await
            .map(|_| ())
    }
}

#[async_trait]
impl BaseMemory for PostgresMessageHistory {
    async fn messages(&self) -> Result<Vec<Message>, MemoryError> {
        self.ensure_table().await?;
        let rows = sqlx::query(AssertSqlSafe(format!(
            "SELECT message FROM {} WHERE session_id = $1 ORDER BY id",
            self.table
        )))
        .bind(&self.session_id)
        .fetch_all(&self.pool)
        .await?;
        decode(
            rows.iter()
                .map(|r| r.try_get::<String, _>("message"))
                .collect::<Result<_, _>>()?,
        )
    }

    async fn add_message(&mut self, message: Message) -> Result<(), MemoryError> {
        self.ensure_table().await?;
        sqlx::query(AssertSqlSafe(format!(
            "INSERT INTO {} (session_id, message) VALUES ($1, $2)",
            self.table
        )))
        .bind(&self.session_id)
        .bind(serde_json::to_string(&message)?)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        self.ensure_table().await?;
        sqlx::query(AssertSqlSafe(format!(
            "DELETE FROM {} WHERE session_id = $1",
            self.table
        )))
        .bind(&self.session_id)
        .execute(&self.pool)
        .await?;
        Ok(())
    }
}

macro_rules! into_shared {
    ($ty:ty) => {
        impl From<$ty> for std::sync::Arc<tokio::sync::Mutex<dyn BaseMemory>> {
            fn from(memory: $ty) -> Self {
                std::sync::Arc::new(tokio::sync::Mutex::new(memory))
            }
        }
    };
}

into_shared!(SqliteMessageHistory);
into_shared!(PostgresMessageHistory);

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use sqlx::sqlite::SqlitePoolOptions;
    use tokio::sync::Mutex;

    use super::*;
    use crate::schemas::MessageType;

    async fn sqlite_pool() -> Pool<Sqlite> {
        SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap()
    }

    #[test]
    fn table_names_are_validated() {
        for ok in ["history", "_h1", "chat.history", "Chat_2"] {
            assert!(validate_table_name(ok).is_ok(), "{ok}");
        }
        for bad in [
            "",
            "1abc",
            "a b",
            "h; DROP TABLE x",
            "a.",
            ".a",
            "h-1",
            "\"h\"",
        ] {
            assert!(validate_table_name(bad).is_err(), "{bad}");
        }
    }

    #[tokio::test]
    async fn sqlite_round_trip_keeps_order_and_fields() {
        let mut memory = SqliteMessageHistory::new(sqlite_pool().await, "s1");
        assert!(memory.messages().await.unwrap().is_empty());

        memory.add_user_message(&"what is 2+2?").await.unwrap();
        memory
            .add_message(Message::new_ai_message("").with_tool_calls(
                json!([{"id": "call_1", "type": "function", "function": {"name": "calc", "arguments": "{}"}}]),
            ))
            .await
            .unwrap();
        memory
            .add_message(Message::new_tool_message("4", "call_1"))
            .await
            .unwrap();
        memory.add_ai_message(&"It is 4").await.unwrap();

        let messages = memory.messages().await.unwrap();
        let types: Vec<_> = messages.iter().map(|m| m.message_type.clone()).collect();
        assert_eq!(
            types,
            vec![
                MessageType::HumanMessage,
                MessageType::AIMessage,
                MessageType::ToolMessage,
                MessageType::AIMessage
            ]
        );
        assert_eq!(messages[0].content, "what is 2+2?");
        assert_eq!(messages[1].tool_calls.as_ref().unwrap()[0]["id"], "call_1");
        assert_eq!(messages[2].id.as_deref(), Some("call_1"));
        assert_eq!(messages[3].content, "It is 4");
    }

    #[tokio::test]
    async fn sqlite_sessions_are_isolated_and_persist_across_instances() {
        let pool = sqlite_pool().await;
        let mut a = SqliteMessageHistory::new(pool.clone(), "a");
        let mut b = SqliteMessageHistory::new(pool.clone(), "b");
        a.add_user_message(&"from a").await.unwrap();
        b.add_user_message(&"from b").await.unwrap();

        // A fresh instance for the same session sees the stored history.
        let reopened = SqliteMessageHistory::new(pool.clone(), "a");
        assert_eq!(reopened.to_string().await.unwrap(), "human: from a");

        a.clear().await.unwrap();
        assert!(a.messages().await.unwrap().is_empty());
        assert_eq!(b.messages().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn sqlite_custom_table_and_bad_table_name() {
        let pool = sqlite_pool().await;
        let mut memory = SqliteMessageHistory::new(pool.clone(), "s").with_table_name("chat_log");
        memory.add_ai_message(&"hi").await.unwrap();
        let count: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM chat_log")
            .fetch_one(&pool)
            .await
            .unwrap();
        assert_eq!(count, 1);

        let mut bad =
            SqliteMessageHistory::new(pool, "s").with_table_name("x; DROP TABLE chat_log");
        let err = bad.add_ai_message(&"hi").await.unwrap_err();
        assert!(matches!(err, MemoryError::Other(_)), "{err:?}");
    }

    #[tokio::test]
    async fn sqlite_without_migration_reports_missing_table() {
        let memory = SqliteMessageHistory::new(sqlite_pool().await, "s").without_migration();
        let err = memory.messages().await.unwrap_err();
        assert!(matches!(err, MemoryError::Storage(_)), "{err:?}");
    }

    #[tokio::test]
    async fn sqlite_history_drives_a_conversational_chain() {
        use crate::{
            chain::{builder::ConversationalChainBuilder, Chain},
            prompt_args,
            test_utils::FakeLLM,
        };

        let pool = sqlite_pool().await;
        let llm = FakeLLM::new(["Hi Luis", "You are Luis"]);
        let memory: Arc<Mutex<dyn BaseMemory>> =
            SqliteMessageHistory::new(pool.clone(), "chat").into();
        let chain = ConversationalChainBuilder::new()
            .llm(llm.clone())
            .memory(memory)
            .build()
            .unwrap();
        chain
            .invoke(prompt_args! {"input" => "I am Luis"})
            .await
            .unwrap();

        // A new chain over the same table remembers the first turn.
        let memory: Arc<Mutex<dyn BaseMemory>> = SqliteMessageHistory::new(pool, "chat").into();
        let chain = ConversationalChainBuilder::new()
            .llm(llm.clone())
            .memory(memory.clone())
            .build()
            .unwrap();
        chain
            .invoke(prompt_args! {"input" => "Who am I?"})
            .await
            .unwrap();

        let second_prompt = format!("{:?}", llm.calls()[1]);
        assert!(second_prompt.contains("I am Luis"), "{second_prompt}");
        assert!(second_prompt.contains("Hi Luis"), "{second_prompt}");
        assert_eq!(memory.lock().await.messages().await.unwrap().len(), 4);
    }

    /// Needs Docker. Run with `cargo test --features postgres -- --ignored postgres_message_history`.
    /// Or point PG_MEMORY_TEST_URL at an existing postgres.
    #[tokio::test]
    #[ignore]
    async fn postgres_message_history_round_trip() {
        let (url, container) = match std::env::var("PG_MEMORY_TEST_URL") {
            Ok(url) => (url, None),
            Err(_) => {
                let out = std::process::Command::new("docker")
                    .args([
                        "run",
                        "-d",
                        "--rm",
                        "-e",
                        "POSTGRES_PASSWORD=pw",
                        "-p",
                        "0:5432",
                        "postgres:17",
                    ])
                    .output()
                    .expect("docker run");
                assert!(out.status.success(), "{out:?}");
                let id = String::from_utf8(out.stdout).unwrap().trim().to_string();
                let port = std::process::Command::new("docker")
                    .args(["port", &id, "5432/tcp"])
                    .output()
                    .unwrap();
                let port = String::from_utf8(port.stdout).unwrap();
                let port = port
                    .lines()
                    .next()
                    .unwrap()
                    .rsplit(':')
                    .next()
                    .unwrap()
                    .to_string();
                (
                    format!("postgres://postgres:pw@127.0.0.1:{port}/postgres"),
                    Some(id),
                )
            }
        };

        let mut pool = None;
        for _ in 0..30 {
            let attempt = sqlx::postgres::PgPoolOptions::new()
                .acquire_timeout(std::time::Duration::from_secs(2))
                .connect(&url)
                .await;
            if let Ok(p) = attempt {
                pool = Some(p);
                break;
            }
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }
        let pool = pool.expect("postgres did not come up (is docker out of disk?)");

        let table = format!("lc_hist_{}", std::process::id());
        let mut a = PostgresMessageHistory::new(pool.clone(), "a").with_table_name(&table);
        let mut b = PostgresMessageHistory::new(pool.clone(), "b").with_table_name(&table);
        a.add_user_message(&"hello").await.unwrap();
        a.add_message(Message::new_ai_message("").with_tool_calls(json!([{"id": "c1"}])))
            .await
            .unwrap();
        a.add_message(Message::new_tool_message("ok", "c1"))
            .await
            .unwrap();
        b.add_user_message(&"other").await.unwrap();

        let reopened = PostgresMessageHistory::new(pool.clone(), "a").with_table_name(&table);
        let messages = reopened.messages().await.unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, "hello");
        assert_eq!(messages[1].tool_calls.as_ref().unwrap()[0]["id"], "c1");
        assert_eq!(messages[2].id.as_deref(), Some("c1"));

        a.clear().await.unwrap();
        assert!(a.messages().await.unwrap().is_empty());
        assert_eq!(b.messages().await.unwrap().len(), 1);

        sqlx::query(AssertSqlSafe(format!("DROP TABLE {table}")))
            .execute(&pool)
            .await
            .unwrap();
        if let Some(id) = container {
            let _ = std::process::Command::new("docker")
                .args(["stop", &id])
                .output();
        }
    }
}
