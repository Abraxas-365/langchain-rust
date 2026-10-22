//! MongoDB Atlas Vector Search (feature `mongodb`).
//!
//! Works with MongoDB Atlas and with the `mongodb/mongodb-atlas-local` image. Documents
//! are stored as `{text, embedding, namespace?, metadata}`; the `vectorSearch` index is
//! created on the first write unless disabled.
//!
//! ```rust,ignore
//! use langchain_rust::vectorstore::mongodb::StoreBuilder;
//!
//! let store = StoreBuilder::new()
//!     .uri("mongodb://localhost:27017/?directConnection=true")
//!     .database("app")
//!     .collection("docs")
//!     .embedder(OpenAiEmbedder::default())
//!     // Paths you want to filter on must be in the index.
//!     .filter_paths(["metadata.source", "metadata.year"])
//!     .build()
//!     .await?;
//! ```
//!
//! Atlas indexes asynchronously: new documents become searchable after a short delay.

mod builder;
mod store;
#[cfg(test)]
mod tests;

pub use builder::*;
pub use store::{MongoSimilarity, Store};
