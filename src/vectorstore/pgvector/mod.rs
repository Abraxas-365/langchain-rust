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

#[cfg(test)]
mod tests {
    use crate::{
        schemas::Document,
        test_utils::FakeEmbedder,
        vectorstore::{pgvector::StoreBuilder, VecStoreOptions, VectorStore},
    };

    /// Needs Docker. Run with `cargo test --features postgres -- --ignored pgvector`.
    /// Or point PGVECTOR_TEST_URL at an existing postgres with the `vector` extension.
    #[tokio::test]
    #[ignore]
    async fn pgvector_similarity_search_returns_closest_first() {
        let url = match std::env::var("PGVECTOR_TEST_URL") {
            Ok(url) => url,
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
                        "pgvector/pgvector:pg17",
                    ])
                    .output()
                    .expect("docker");
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
                format!("postgres://postgres:pw@127.0.0.1:{port}/postgres")
            }
        };

        let embedder = FakeEmbedder::new(
            3,
            &[
                ("near", vec![1.0, 0.0, 0.0]),
                ("mid", vec![1.0, 1.0, 0.0]),
                ("far", vec![-1.0, 0.0, 0.0]),
                ("query", vec![1.0, 0.05, 0.0]),
            ],
        );

        let mut store = None;
        for _ in 0..60 {
            match StoreBuilder::new()
                .embedder(embedder.clone())
                .connection_url(&url)
                .vector_dimensions(3)
                .collection_name("order_test")
                .pre_delete_collection(true)
                .build()
                .await
            {
                Ok(s) => {
                    store = Some(s);
                    break;
                }
                Err(_) => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
            }
        }
        let store = store.expect("postgres did not come up");

        let docs = ["far", "mid", "near"].map(Document::new);
        store
            .add_documents(&docs, &VecStoreOptions::default())
            .await
            .unwrap();

        let found = store
            .similarity_search("query", 2, &VecStoreOptions::default())
            .await
            .unwrap();
        let contents: Vec<_> = found.iter().map(|d| d.page_content.as_str()).collect();
        assert_eq!(contents, ["near", "mid"]);
        assert!(found[0].score <= found[1].score, "score is cosine distance");
    }
}
