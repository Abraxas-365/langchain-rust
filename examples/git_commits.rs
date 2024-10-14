// To run this example execute: cargo run --example git_commits --features sqlite-vss,git -- /path/to/git/repo
// Make sure vector0 and vss0 libraries are installed in the system or the path of the executable.
// Download the libraries from https://github.com/asg017/sqlite-vss
// For static compilation of sqlite-vss extension refer to the following link:
// https://github.com/launchbadge/sqlx/issues/3147.

#[cfg(feature = "sqlite-vss")]
use futures_util::StreamExt;
#[cfg(feature = "sqlite-vss")]
use langchain_rust::{
    document_loaders::GitCommitLoader,
    document_loaders::Loader,
    embedding::openai::OpenAiEmbedder,
    vectorstore::{sqlite_vss::StoreBuilder, VecStoreOptions, VectorStore},
};
#[cfg(feature = "sqlite-vss")]
use std::io::Write;

#[cfg(feature = "sqlite-vss")]
#[tokio::main]
async fn main() {
    // Initialize Embedder
    let embedder = OpenAiEmbedder::default();

    let database_url = std::env::var("DATABASE_URL").unwrap_or("sqlite::memory:".to_string());

    let repo_path = std::env::args()
        .nth(1)
        .expect("Please provide the path to the git repository.");

    // Initialize the Sqlite Vector Store
    let store = StoreBuilder::new()
        .embedder(embedder)
        .connection_url(database_url)
        .table("documents")
        .vector_dimensions(1536)
        .build()
        .await
        .unwrap();

    // Initialize the tables in the database. This is required to be done only once.
    store.initialize().await.unwrap();

    let git_commit_loader = GitCommitLoader::from_path(repo_path).unwrap();

    let mut stream = git_commit_loader.load().await.unwrap();
    while let Some(result) = stream.next().await {
        match result {
            Ok(document) => {
                store
                    .add_documents(&[document], &VecStoreOptions::default())
                    .await
                    .unwrap();
            }
            Err(e) => panic!("Error fetching git commits {:?}", e),
        }
    }

    // Ask for user input
    print!("Query> ");
    std::io::stdout().flush().unwrap();
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    let results = store
        .similarity_search(&input, 2, &VecStoreOptions::default())
        .await
        .unwrap();

    if results.is_empty() {
        println!("No results found.");
        return;
    } else {
        results.iter().for_each(|r| {
            println!("Document: {}", r.page_content);
        });
    }
}

#[cfg(not(feature = "sqlite-vss"))]
fn main() {
    println!("This example requires the 'sqlite-vss' and 'git' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example git_commits --features sqlite-vss,git -- /path/to/git/repo");
}
