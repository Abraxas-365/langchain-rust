// To run this example execute: cargo run --example git_commits --features sqlite,git -- /path/to/git/repo
// Make sure vector0 and vss0 libraries are installed in the system or the path of the executable.
// Download the libraries from https://github.com/asg017/sqlite-vss
// For static compliation of sqlite-vss extension refer to the following link:
// https://github.com/launchbadge/sqlx/issues/3147.

#[cfg(feature = "sqlite")]
use futures_util::StreamExt;
#[cfg(feature = "sqlite")]
use langchain_rust::{
    document_loaders::GitCommitLoader,
    document_loaders::Loader,
    embedding::openai::OpenAiEmbedder,
    vectorstore::{sqlite_vss::StoreBuilder, VecStoreOptions, VectorStore},
};
#[cfg(feature = "sqlite")]
use std::io::Write;

#[cfg(feature = "sqlite")]
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

    // Intialize the tables in the database. This is required to be done only once.
    store.initialize().await.unwrap();

    let git_commit_loader = GitCommitLoader::from_path(repo_path).unwrap();

    let documents = git_commit_loader
        .load()
        .await
        .unwrap()
        .map(|x| x.unwrap())
        .collect::<Vec<_>>()
        .await;

    store
        .add_documents(&documents, &VecStoreOptions::default())
        .await
        .unwrap();

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

#[cfg(not(feature = "sqlite"))]
fn main() {
    println!("This example requires the 'sqlite' and 'git' feature to be enabled.");
    println!("Please run the command as follows:");
    println!("cargo run --example git_commits --features sqlite,git -- /path/to/git/repo");
}
