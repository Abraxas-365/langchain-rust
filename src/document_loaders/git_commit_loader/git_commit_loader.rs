use std::collections::HashMap;
use std::pin::Pin;

use crate::document_loaders::{process_doc_stream, LoaderError};
use crate::{document_loaders::Loader, schemas::Document, text_splitter::TextSplitter};
use async_trait::async_trait;
use futures::Stream;
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

        // Since commits_iter can't be shared across thread safely, use channels as a workaround.
        let (tx, rx) = flume::bounded(1);

        // The gix walk and the bounded `send` are blocking: run them off the async runtime
        // (with `tokio::spawn` a current-thread runtime deadlocks against the receiver).
        tokio::task::spawn_blocking(move || {
            let walk = repo
                .head_id()
                .map_err(|e| LoaderError::OtherError(format!("git HEAD: {e}")))
                .and_then(|head| {
                    repo.rev_walk(Some(head.detach()))
                        .all()
                        .map_err(|e| LoaderError::OtherError(format!("git rev walk: {e}")))
                });
            let commits = match walk {
                Ok(commits) => commits,
                Err(err) => {
                    let _ = tx.send(Err(err));
                    return;
                }
            };

            for info in commits.filter_map(Result::ok) {
                let document = commit_document(&info);
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

fn commit_document(info: &gix::revision::walk::Info<'_>) -> Result<Document, LoaderError> {
    let to_err = |what: &str, e: &dyn std::fmt::Display| {
        LoaderError::OtherError(format!("git {what} of {}: {e}", info.id))
    };
    let commit = info.object().map_err(|e| to_err("object", &e))?;
    let author = commit.author().map_err(|e| to_err("author", &e))?;
    let message = commit.message().map_err(|e| to_err("message", &e))?;
    let commit_id = commit.id;

    let mut document = Document::new(format!(
        "commit {commit_id}\nAuthor: {} <{}>\n\n    {}",
        author.name, author.email, message.title
    ));
    document.metadata = HashMap::from([("commit".to_string(), Value::from(commit_id.to_string()))]);
    Ok(document)
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use futures_util::StreamExt;

    use super::*;

    fn git(dir: &std::path::Path, args: &[&str], date: &str) {
        let status = Command::new("git")
            .args(args)
            .current_dir(dir)
            .env("GIT_AUTHOR_NAME", "Ada")
            .env("GIT_AUTHOR_EMAIL", "ada@example.com")
            .env("GIT_COMMITTER_NAME", "Ada")
            .env("GIT_COMMITTER_EMAIL", "ada@example.com")
            .env("GIT_AUTHOR_DATE", date)
            .env("GIT_COMMITTER_DATE", date)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .status()
            .expect("git must be installed");
        assert!(status.success(), "git {args:?} failed");
    }

    #[tokio::test]
    async fn loads_commits_from_a_local_repo() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("lcr-git-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let t1 = "2020-01-01T00:00:00Z";
        git(&dir, &["init", "-q", "-b", "main"], t1);
        git(
            &dir,
            &["commit", "-q", "--allow-empty", "-m", "first commit"],
            t1,
        );
        git(
            &dir,
            &[
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "second commit",
                "-m",
                "body",
            ],
            "2020-01-02T00:00:00Z",
        );

        let documents = GitCommitLoader::from_path(&dir)
            .unwrap()
            .load()
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await;
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(documents.len(), 2, "{documents:?}");
        // Newest first.
        assert!(documents[0]
            .page_content
            .trim_end()
            .ends_with("Author: Ada <ada@example.com>\n\n    second commit"));
        assert!(documents[1]
            .page_content
            .trim_end()
            .ends_with("    first commit"));
        let commit = documents[0].metadata["commit"].as_str().unwrap();
        assert_eq!(commit.len(), 40);
        assert!(documents[0]
            .page_content
            .starts_with(&format!("commit {commit}\n")));
    }

    #[tokio::test]
    async fn empty_repo_yields_an_error_instead_of_panicking() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("lcr-git-empty-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        git(&dir, &["init", "-q", "-b", "main"], "2020-01-01T00:00:00Z");

        let results = GitCommitLoader::from_path(&dir)
            .unwrap()
            .load()
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(results.len(), 1);
        assert!(results[0].is_err());
    }
}
