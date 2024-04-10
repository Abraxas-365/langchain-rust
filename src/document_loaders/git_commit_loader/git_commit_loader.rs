use std::collections::HashMap;
use std::pin::Pin;

use crate::document_loaders::{process_doc_stream, LoaderError};
use crate::{document_loaders::Loader, schemas::Document, text_splitter::TextSplitter};
use async_trait::async_trait;
use futures::Stream;
use futures_util::stream;
use gix::ThreadSafeRepository;
use serde_json::Value;

#[derive(Clone)]
pub struct GitCommitLoader {
    repo: ThreadSafeRepository,
}

impl GitCommitLoader {
    pub fn new(repo: ThreadSafeRepository) -> Self {
        Self { repo }
    }

    pub fn from_path<P: AsRef<std::path::Path>>(directory: P) -> Result<Self, LoaderError> {
        let repo = ThreadSafeRepository::discover(directory)?;
        Ok(Self::new(repo))
    }
}

#[async_trait]
impl Loader for GitCommitLoader {
    async fn load(
        mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let repo = self.repo.to_thread_local();

        let revwalk = repo
            .rev_walk(Some(repo.head_id().unwrap().detach()))
            .all()
            .unwrap()
            .filter_map(Result::ok);

        let commits_iter = revwalk.map(|oid| {
            let commit = oid.object().unwrap();
            let commit_id = commit.id;
            let author = commit.author().unwrap();
            let email = author.email.to_string();
            let name = author.name.to_string();
            let message = format!("{}", commit.message().unwrap().title);

            let mut document = Document::new(format!(
                "commit {commit_id}\nAuthor: {name} <{email}>\n\n    {message}"
            ));
            let mut metadata = HashMap::new();
            metadata.insert("commit".to_string(), Value::from(commit_id.to_string()));

            document.metadata = metadata;
            Ok(document)
        });

        // TODO: This is a temporary solution to collect all the docs as can't share it between threads
        let documents = commits_iter.collect::<Vec<_>>();

        Ok(Box::pin(stream::iter(documents)))
    }

    async fn load_and_split<TS: TextSplitter + 'static>(
        mut self,
        splitter: TS,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let doc_stream = self.load().await?;
        let stream = process_doc_stream(doc_stream, splitter).await;
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use futures_util::StreamExt;

    use super::*;

    #[tokio::test]
    #[ignore]
    async fn git_commit_loader() {
        let git_commit_loader = GitCommitLoader::from_path("/code/langchain-rust").unwrap();

        let documents = git_commit_loader
            .load()
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await;

        dbg!(&documents);
        // assert_eq!(documents[0].page_content, "");
        todo!()
    }
}
