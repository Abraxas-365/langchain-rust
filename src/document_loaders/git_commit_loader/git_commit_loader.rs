use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use crate::document_loaders::{process_doc_stream, LoaderError};
use crate::{document_loaders::Loader, schemas::Document, text_splitter::TextSplitter};
use async_trait::async_trait;
use futures::Stream;
use gix::ThreadSafeRepository;
use serde_json::Value;

/// A commit as seen by [`GitCommitLoader::with_filter`] and
/// [`GitCommitLoader::with_document_builder`].
#[derive(Debug, Clone, PartialEq)]
pub struct GitCommit {
    /// Full hex object id.
    pub id: String,
    /// Parent ids, first parent first. Empty for a root commit.
    pub parent_ids: Vec<String>,
    pub author_name: String,
    pub author_email: String,
    /// Author time, seconds since the UNIX epoch.
    pub author_time: i64,
    /// Committer time, seconds since the UNIX epoch.
    pub commit_time: i64,
    /// First paragraph of the message.
    pub summary: String,
    /// The full message.
    pub message: String,
}

type CommitFilter = Arc<dyn Fn(&GitCommit) -> bool + Send + Sync>;
type DocumentBuilder = Arc<dyn Fn(&GitCommit) -> Document + Send + Sync>;

/// Loads the commits reachable from `HEAD`, newest first, one [`Document`] per commit.
///
/// ```rust,ignore
/// let loader = GitCommitLoader::from_path(".")?
///     // only what's new since the last run
///     .since(&last_indexed_commit)
///     .with_filter(|c| !c.summary.starts_with("Merge "));
/// ```
#[derive(Clone)]
pub struct GitCommitLoader {
    repo: ThreadSafeRepository,
    since: Option<String>,
    filter: Option<CommitFilter>,
    document_builder: Option<DocumentBuilder>,
}

impl GitCommitLoader {
    pub fn new(repo: ThreadSafeRepository) -> Self {
        Self {
            repo,
            since: None,
            filter: None,
            document_builder: None,
        }
    }

    pub fn from_path<P: AsRef<std::path::Path>>(directory: P) -> Result<Self, LoaderError> {
        let repo = ThreadSafeRepository::discover(directory)?;
        Ok(Self::new(repo))
    }

    /// Skips `rev` and everything reachable from it, like `git log <rev>..HEAD`. `rev` is any
    /// revision git understands (a commit id, a tag, a branch). Handy for indexing only the
    /// commits added since the last run. Loading fails if `rev` doesn't resolve.
    pub fn since<S: Into<String>>(mut self, rev: S) -> Self {
        self.since = Some(rev.into());
        self
    }

    /// Loads only the commits for which `filter` returns `true`.
    pub fn with_filter<F>(mut self, filter: F) -> Self
    where
        F: Fn(&GitCommit) -> bool + Send + Sync + 'static,
    {
        self.filter = Some(Arc::new(filter));
        self
    }

    /// Builds each document with `builder` instead of the default
    /// `commit <id>\nAuthor: <name> <email>\n\n    <summary>` text.
    pub fn with_document_builder<F>(mut self, builder: F) -> Self
    where
        F: Fn(&GitCommit) -> Document + Send + Sync + 'static,
    {
        self.document_builder = Some(Arc::new(builder));
        self
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
        let since = self.since.take();
        let filter = self.filter.take();
        let document_builder = self.document_builder.take();

        // Since commits_iter can't be shared across thread safely, use channels as a workaround.
        let (tx, rx) = flume::bounded(1);

        // The gix walk and the bounded `send` are blocking: run them off the async runtime
        // (with `tokio::spawn` a current-thread runtime deadlocks against the receiver).
        tokio::task::spawn_blocking(move || {
            let hidden = match since.as_deref().map(|rev| resolve_commit(&repo, rev)) {
                None => None,
                Some(Ok(id)) => Some(id),
                Some(Err(err)) => {
                    let _ = tx.send(Err(err));
                    return;
                }
            };
            let walk = repo
                .head_id()
                .map_err(|e| LoaderError::OtherError(format!("git HEAD: {e}")))
                .and_then(|head| {
                    repo.rev_walk(Some(head.detach()))
                        .with_hidden(hidden)
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
                let document = match read_commit(&info) {
                    Ok(commit) => {
                        if filter.as_ref().is_some_and(|keep| !keep(&commit)) {
                            continue;
                        }
                        Ok(match &document_builder {
                            Some(build) => build(&commit),
                            None => default_document(&commit),
                        })
                    }
                    Err(err) => Err(err),
                };
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

fn resolve_commit(repo: &gix::Repository, rev: &str) -> Result<gix::ObjectId, LoaderError> {
    let err = |e: &dyn std::fmt::Display| LoaderError::OtherError(format!("git rev `{rev}`: {e}"));
    let id = repo.rev_parse_single(rev).map_err(|e| err(&e))?;
    let object = id.object().map_err(|e| err(&e))?;
    Ok(object.peel_to_commit().map_err(|e| err(&e))?.id)
}

fn read_commit(info: &gix::revision::walk::Info<'_>) -> Result<GitCommit, LoaderError> {
    let to_err = |what: &str, e: &dyn std::fmt::Display| {
        LoaderError::OtherError(format!("git {what} of {}: {e}", info.id))
    };
    let commit = info.object().map_err(|e| to_err("object", &e))?;
    let author = commit.author().map_err(|e| to_err("author", &e))?;
    let author_time = author.time().map_err(|e| to_err("author time", &e))?;
    let commit_time = commit.time().map_err(|e| to_err("commit time", &e))?;
    let message = commit.message().map_err(|e| to_err("message", &e))?;
    let raw_message = commit.message_raw().map_err(|e| to_err("message", &e))?;

    Ok(GitCommit {
        id: commit.id.to_string(),
        parent_ids: commit.parent_ids().map(|id| id.to_string()).collect(),
        author_name: author.name.to_string(),
        author_email: author.email.to_string(),
        author_time: author_time.seconds,
        commit_time: commit_time.seconds,
        summary: message.summary().to_string(),
        message: raw_message.to_string(),
    })
}

fn default_document(commit: &GitCommit) -> Document {
    let mut document = Document::new(format!(
        "commit {}\nAuthor: {} <{}>\n\n    {}",
        commit.id, commit.author_name, commit.author_email, commit.summary
    ));
    document.metadata = HashMap::from([("commit".to_string(), Value::from(commit.id.clone()))]);
    document
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

    fn git_out(dir: &std::path::Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .args(args)
            .current_dir(dir)
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .output()
            .expect("git must be installed");
        assert!(out.status.success(), "git {args:?} failed");
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    }

    /// Three commits, oldest first: `first commit` (annotated tag v1), `feat: second` (with
    /// a body and a committer date later than its author date) and `chore: third` (subject
    /// wrapped over two lines). Returns the dir and the commit ids, oldest first.
    fn fixture(name: &str) -> (std::path::PathBuf, Vec<String>) {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("lcr-git-{name}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let t1 = "2020-01-01T00:00:00Z";
        git(&dir, &["init", "-q", "-b", "main"], t1);
        git(
            &dir,
            &["commit", "-q", "--allow-empty", "-m", "first commit"],
            t1,
        );
        // Annotated, so `since("v1")` has to peel the tag object to its commit.
        git(&dir, &["tag", "-a", "v1", "-m", "release 1"], t1);
        let status = Command::new("git")
            .args([
                "commit",
                "-q",
                "--allow-empty",
                "-m",
                "feat: second",
                "-m",
                "the body",
            ])
            .current_dir(&dir)
            .env("GIT_AUTHOR_NAME", "Grace")
            .env("GIT_AUTHOR_EMAIL", "grace@example.com")
            .env("GIT_COMMITTER_NAME", "Ada")
            .env("GIT_COMMITTER_EMAIL", "ada@example.com")
            .env("GIT_AUTHOR_DATE", "2020-01-02T00:00:00Z")
            .env("GIT_COMMITTER_DATE", "2020-01-03T00:00:00Z")
            .env("GIT_CONFIG_GLOBAL", "/dev/null")
            .env("GIT_CONFIG_NOSYSTEM", "1")
            .status()
            .unwrap();
        assert!(status.success());
        // A subject wrapped over two lines: the summary joins them, like `git log --format=%s`.
        git(
            &dir,
            &["commit", "-q", "--allow-empty", "-m", "chore:\nthird"],
            "2020-01-04T00:00:00Z",
        );
        let ids = ["HEAD~2", "HEAD~1", "HEAD"]
            .iter()
            .map(|rev| git_out(&dir, &["rev-parse", rev]))
            .collect();
        (dir, ids)
    }

    async fn summaries(loader: GitCommitLoader) -> Vec<String> {
        loader
            .with_document_builder(|c| Document::new(c.summary.clone()))
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap().page_content)
            .collect()
            .await
    }

    #[tokio::test]
    async fn document_builder_sees_every_commit_field() {
        let (dir, ids) = fixture("builder");
        let commits = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen = commits.clone();
        let documents = GitCommitLoader::from_path(&dir)
            .unwrap()
            .with_document_builder(move |c| {
                seen.lock().unwrap().push(c.clone());
                let mut doc = Document::new(format!("{}: {}", c.author_name, c.summary));
                doc.metadata.insert("sha".into(), Value::from(c.id.clone()));
                doc
            })
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;
        let _ = std::fs::remove_dir_all(&dir);

        // The builder's documents are used as they are.
        let contents: Vec<_> = documents.iter().map(|d| d.page_content.as_str()).collect();
        assert_eq!(
            contents,
            [
                "Ada: chore: third",
                "Grace: feat: second",
                "Ada: first commit"
            ]
        );
        assert_eq!(documents[1].metadata.len(), 1);
        assert_eq!(documents[1].metadata["sha"], Value::from(ids[1].clone()));

        let commits = commits.lock().unwrap();
        assert_eq!(
            commits[1],
            GitCommit {
                id: ids[1].clone(),
                parent_ids: vec![ids[0].clone()],
                author_name: "Grace".into(),
                author_email: "grace@example.com".into(),
                author_time: 1_577_923_200,
                commit_time: 1_578_009_600,
                summary: "feat: second".into(),
                message: "feat: second\n\nthe body\n".into(),
            }
        );
        assert!(commits[2].parent_ids.is_empty(), "root commit");
        assert_eq!(commits[0].parent_ids, vec![ids[1].clone()]);
    }

    #[tokio::test]
    async fn filter_keeps_only_matching_commits_in_order() {
        let (dir, _) = fixture("filter");
        let loader = GitCommitLoader::from_path(&dir).unwrap();
        let kept = summaries(loader.clone().with_filter(|c| c.author_name == "Ada")).await;
        let none = summaries(loader.clone().with_filter(|_| false)).await;
        // The default document format is kept when only a filter is set.
        let default_docs = loader
            .with_filter(|c| c.summary.starts_with("feat"))
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(kept, ["chore: third", "first commit"]);
        assert!(none.is_empty());
        assert_eq!(default_docs.len(), 1);
        assert!(default_docs[0]
            .page_content
            .ends_with("Author: Grace <grace@example.com>\n\n    feat: second"));
    }

    #[tokio::test]
    async fn since_skips_what_was_already_loaded() {
        let (dir, ids) = fixture("since");
        let loader = GitCommitLoader::from_path(&dir).unwrap();
        let after_first = summaries(loader.clone().since(ids[0].clone())).await;
        let after_tag = summaries(loader.clone().since("v1")).await;
        let after_head = summaries(loader.clone().since("HEAD")).await;
        let with_filter = summaries(
            loader
                .clone()
                .since(&ids[0][..10])
                .with_filter(|c| c.summary.starts_with("chore")),
        )
        .await;
        let unknown = loader
            .since("no-such-branch")
            .load()
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
        let _ = std::fs::remove_dir_all(&dir);

        assert_eq!(after_first, ["chore: third", "feat: second"]);
        assert_eq!(after_tag, after_first, "tags resolve to their commit");
        assert!(after_head.is_empty());
        assert_eq!(with_filter, ["chore: third"]);
        assert_eq!(unknown.len(), 1);
        let err = unknown[0].as_ref().unwrap_err().to_string();
        assert!(err.contains("no-such-branch"), "{err}");
    }
}
