use std::collections::HashMap;
use std::marker::PhantomData;
use std::pin::Pin;

use crate::document_loaders::{process_doc_stream, LoaderError};
use crate::{document_loaders::Loader, schemas::Document, text_splitter::TextSplitter};
use async_trait::async_trait;
use futures::Stream;
use gix::revision::walk::Info;
use gix::ThreadSafeRepository;
use serde_json::Value;

#[derive(Clone)]
pub struct GitCommitLoader<F, M, T> {
    repo: ThreadSafeRepository,
    filter: Option<F>,
    map: Option<M>,
    resource_type: PhantomData<T>,
}

impl<F, M, T> GitCommitLoader<F, M, T> {
    pub fn new(repo: ThreadSafeRepository) -> Self {
        Self {
            repo,
            filter: None,
            map: None,
            resource_type: PhantomData::<T>,
        }
    }

    pub fn from_path<P: AsRef<std::path::Path>>(directory: P) -> Result<Self, LoaderError> {
        let repo = ThreadSafeRepository::discover(directory)?;
        Ok(Self::new(repo))
    }

    pub fn with_filter(mut self, filter: F) -> Self
    where
        F: Fn(&Info) -> bool,
    {
        self.filter = Some(filter);
        self
    }

    pub fn with_map(mut self, map: M) -> Self
    where
        M: Fn(&Info) -> Result<Document, LoaderError>,
    {
        self.map = Some(map);
        self
    }
}

#[async_trait]
impl<
        F: Send + Sync + 'static + Copy + for<'a, 'b> Fn(&'a Info<'b>) -> bool,
        M: Send
            + Sync
            + 'static
            + Copy
            + for<'a, 'b> Fn(&'a Info<'b>) -> Result<Document, LoaderError>,
        T: Send + Sync,
    > Loader<F, M, T> for GitCommitLoader<F, M, T>
{
    async fn load(
        self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let repo = self.repo.to_thread_local();

        // Since commits_iter can't be shared across thread safely, use channels as a workaround.
        let (tx, rx) = flume::bounded(1);

        tokio::spawn(async move {
            let commits_iter = repo
                .rev_walk(Some(repo.head_id().unwrap().detach()))
                .all()
                .unwrap()
                .map(|x| x.unwrap())
                .filter(|x| {
                    if let Some(f) = self.filter {
                        f(&x)
                    } else {
                        true
                    }
                })
                .map(|oid| {
                    if let Some(m) = self.map {
                        m(&oid)
                    } else {
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
                    }
                });

            for document in commits_iter {
                if tx.send(document).is_err() {
                    // stream might have been dropped
                    break;
                }
            }
        });

        Ok(Box::pin(rx.into_stream()))
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
        let username = "langchain".to_string();
        let git_commit_loader = GitCommitLoader::from_path("/code/langchain-rust")
            .unwrap()
            .with_filter(move |info| info.object().unwrap().author().unwrap().name == username);

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
