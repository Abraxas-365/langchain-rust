//! Redis vector store (feature `redis`), on the Redis Query Engine (RediSearch).
//!
//! Needs Redis 8 (the query engine is built in) or Redis Stack. Documents are hashes at
//! `{key_prefix}{uuid}` holding the text, the embedding as FLOAT32 bytes, the namespace as
//! a TAG, and the metadata. The index is created on the first write.
//!
//! ```rust,ignore
//! use langchain_rust::vectorstore::redis::{MetadataField, StoreBuilder};
//!
//! let store = StoreBuilder::new()
//!     .url("redis://127.0.0.1/")
//!     .index_name("docs")
//!     .embedder(OpenAiEmbedder::default())
//!     // Metadata you want to filter on must be in the index.
//!     .metadata_schema([MetadataField::tag("source"), MetadataField::numeric("year")])
//!     .build()
//!     .await?;
//! ```

mod builder;
mod store;
#[cfg(test)]
mod tests;

pub use builder::*;
pub use store::{FieldType, MetadataField, RedisDistance, Store, VectorAlgorithm};
