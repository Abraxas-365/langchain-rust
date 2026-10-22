mod options;

#[cfg(feature = "postgres")]
pub mod pgvector;

#[cfg(feature = "redis")]
pub mod redis;

#[cfg(feature = "sqlite-vss")]
pub mod sqlite_vss;

#[cfg(feature = "sqlite-vec")]
pub mod sqlite_vec;

#[cfg(feature = "surrealdb")]
pub mod surrealdb;

#[cfg(feature = "weaviate")]
pub mod weaviate;

#[cfg(feature = "opensearch")]
pub mod opensearch;

#[cfg(feature = "qdrant")]
pub mod qdrant;

#[cfg(feature = "mongodb")]
pub mod mongodb;

#[cfg(feature = "chroma")]
pub mod chroma;

#[cfg(feature = "pinecone")]
pub mod pinecone;

mod vectorstore;

pub use options::*;
pub use vectorstore::*;
