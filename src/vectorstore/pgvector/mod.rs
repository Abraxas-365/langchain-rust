mod builder;
mod pgvector;

pub use builder::*;
pub use pgvector::*;

// pgLockIDEmbeddingTable is used for advisor lock to fix issue arising from concurrent
// creation of the embedding table.The same value represents the same lock.
const PG_LOCK_ID_EMBEDDING_TABLE: i64 = 1573678846307946494;
// pgLockIDCollectionTable is used for advisor lock to fix issue arising from concurrent
// creation of the collection table.The same value represents the same lock.
const PG_LOCK_ID_COLLECTION_TABLE: i64 = 1573678846307946495;
// pgLockIDExtension is used for advisor lock to fix issue arising from concurrent creation
// of the vector extension. The value is deliberately set to the same as python langchain
// https://github.com/langchain-ai/langchain/blob/v0.0.340/libs/langchain/langchain/vectorstores/pgvector.py#L167
const PG_LOCKID_EXTENSION: i64 = 1573678846307946496;
