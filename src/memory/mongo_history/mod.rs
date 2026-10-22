//! Persistent chat history in MongoDB (feature `mongodb`).
//!
//! Each message is one document, `{ session_id, message, created_at }`, where `message`
//! is the message as an embedded document (tool calls, tool call ids and images
//! included). Messages come back in insertion order, sorted by `_id`. `created_at` is a
//! BSON date, so a TTL index can expire old messages:
//! `db.message_store.createIndex({ created_at: 1 }, { expireAfterSeconds: 86400 })`.
//!
//! This works with any MongoDB deployment; Atlas isn't required.

use async_trait::async_trait;
use futures::TryStreamExt;
use mongodb::{
    bson::{doc, from_document, to_document, DateTime, Document as BsonDocument},
    options::{FindOptions, IndexOptions},
    Client, Collection, IndexModel,
};
use tokio::sync::OnceCell;

use crate::schemas::{
    memory::{BaseMemory, MemoryError},
    messages::Message,
};

pub const DEFAULT_MONGO_HISTORY_DATABASE: &str = "chat_history";
pub const DEFAULT_MONGO_HISTORY_COLLECTION: &str = "message_store";
const SESSION_ID: &str = "session_id";
const MESSAGE: &str = "message";
const CREATED_AT: &str = "created_at";
const INDEX_NAME: &str = "session_id_1__id_1";

/// Chat history stored in a MongoDB collection.
///
/// ```rust,ignore
/// let memory = MongoMessageHistory::from_uri("mongodb://localhost:27017", "user-42").await?;
/// let chain = ConversationalChainBuilder::new().llm(llm).memory(memory.into()).build()?;
/// ```
///
/// Many sessions can share one collection. The first write creates an index on
/// `{ session_id: 1, _id: 1 }` (skip that with [`without_index`](Self::without_index)).
pub struct MongoMessageHistory {
    collection: Collection<BsonDocument>,
    session_id: String,
    create_index: bool,
    index_ready: OnceCell<()>,
}

impl MongoMessageHistory {
    /// Connects to `uri` and uses the `chat_history.message_store` collection (see
    /// [`with_collection`](Self::with_collection) to pick another one).
    pub async fn from_uri(uri: &str, session_id: impl Into<String>) -> Result<Self, MemoryError> {
        let client = Client::with_uri_str(uri).await?;
        let collection = client
            .database(DEFAULT_MONGO_HISTORY_DATABASE)
            .collection(DEFAULT_MONGO_HISTORY_COLLECTION);
        Ok(Self::new(collection, session_id))
    }

    /// Uses an existing collection handle, e.g.
    /// `client.database("app").collection("chat")`.
    pub fn new(collection: Collection<BsonDocument>, session_id: impl Into<String>) -> Self {
        Self {
            collection,
            session_id: session_id.into(),
            create_index: true,
            index_ready: OnceCell::new(),
        }
    }

    /// Moves to `database.collection` on the same client.
    pub fn with_collection(mut self, database: &str, collection: &str) -> Self {
        self.collection = self
            .collection
            .client()
            .database(database)
            .collection(collection);
        self.index_ready = OnceCell::new();
        self
    }

    /// Don't create the `{ session_id: 1, _id: 1 }` index (for read-only users, or when
    /// you manage indexes yourself).
    pub fn without_index(mut self) -> Self {
        self.create_index = false;
        self
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    pub fn collection(&self) -> &Collection<BsonDocument> {
        &self.collection
    }

    pub(crate) fn session_filter(&self) -> BsonDocument {
        doc! { SESSION_ID: &self.session_id }
    }

    pub(crate) fn index_model() -> IndexModel {
        IndexModel::builder()
            .keys(doc! { SESSION_ID: 1, "_id": 1 })
            .options(IndexOptions::builder().name(INDEX_NAME.to_string()).build())
            .build()
    }

    /// Insertion order (`_id` is an increasing ObjectId), only the message.
    pub(crate) fn find_options() -> FindOptions {
        FindOptions::builder()
            .sort(doc! { "_id": 1 })
            .projection(doc! { MESSAGE: 1 })
            .build()
    }

    pub(crate) fn to_stored(&self, message: &Message) -> Result<BsonDocument, MemoryError> {
        let message = to_document(message)
            .map_err(|e| MemoryError::Other(format!("can't convert the message to BSON: {e}")))?;
        Ok(doc! {
            SESSION_ID: &self.session_id,
            MESSAGE: message,
            CREATED_AT: DateTime::now(),
        })
    }

    pub(crate) fn from_stored(stored: &BsonDocument) -> Result<Message, MemoryError> {
        let message = stored.get_document(MESSAGE).map_err(|e| {
            MemoryError::Other(format!(
                "stored chat message has no `{MESSAGE}` document: {e}"
            ))
        })?;
        from_document(message.clone())
            .map_err(|e| MemoryError::Other(format!("invalid stored chat message: {e}")))
    }

    async fn ensure_index(&self) -> Result<(), MemoryError> {
        if !self.create_index {
            return Ok(());
        }
        self.index_ready
            .get_or_try_init(|| async {
                self.collection.create_index(Self::index_model()).await?;
                Ok::<_, MemoryError>(())
            })
            .await?;
        Ok(())
    }
}

#[async_trait]
impl BaseMemory for MongoMessageHistory {
    async fn messages(&self) -> Result<Vec<Message>, MemoryError> {
        let stored: Vec<BsonDocument> = self
            .collection
            .find(self.session_filter())
            .with_options(Self::find_options())
            .await?
            .try_collect()
            .await?;
        stored.iter().map(Self::from_stored).collect()
    }

    async fn add_message(&mut self, message: Message) -> Result<(), MemoryError> {
        self.ensure_index().await?;
        self.collection
            .insert_one(self.to_stored(&message)?)
            .await?;
        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        self.collection.delete_many(self.session_filter()).await?;
        Ok(())
    }
}

impl From<MongoMessageHistory> for std::sync::Arc<tokio::sync::Mutex<dyn BaseMemory>> {
    fn from(memory: MongoMessageHistory) -> Self {
        std::sync::Arc::new(tokio::sync::Mutex::new(memory))
    }
}

#[cfg(test)]
mod tests;
