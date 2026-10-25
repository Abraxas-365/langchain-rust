//! Pages from a Notion workspace export (langchaingo's `NotionDirectoryLoader`).
//!
//! In Notion: Settings → Export → "Markdown & CSV", unzip, and point the loader at the
//! folder. Each page becomes one document. Notion stores subpages in a folder next to the
//! parent page, so the folder is walked recursively (langchaingo only reads the top level).
//!
//! ```no_run
//! use futures_util::StreamExt;
//! use langchain_rust::document_loaders::{Loader, NotionDirectoryLoader};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let mut docs = NotionDirectoryLoader::new("./Export-1f2e3d4c").load().await?;
//! while let Some(doc) = docs.next().await {
//!     let doc = doc?;
//!     println!("{}: {} chars", doc.metadata["title"], doc.page_content.len());
//! }
//! # Ok(())
//! # }
//! ```

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    pin::Pin,
};

use async_trait::async_trait;
use futures::{stream, Stream};
use serde_json::{json, Value};

use crate::{
    document_loaders::{process_doc_stream, Loader, LoaderError},
    schemas::Document,
    text_splitter::TextSplitter,
};

/// Loads every `.md` page under a Notion export folder, in path order.
///
/// Metadata: `source` (the file path), `title` (the file name without the extension and
/// without the 32-hex page id Notion appends) and `notion_id` (that id, when present).
/// Symlinks are not followed.
#[derive(Debug, Clone)]
pub struct NotionDirectoryLoader {
    root: PathBuf,
}

impl NotionDirectoryLoader {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Reads all pages.
    pub async fn load_documents(&self) -> Result<Vec<Document>, LoaderError> {
        let mut pages = Vec::new();
        let mut pending = vec![self.root.clone()];
        while let Some(dir) = pending.pop() {
            let mut entries = tokio::fs::read_dir(&dir).await?;
            while let Some(entry) = entries.next_entry().await? {
                let file_type = entry.file_type().await?;
                let path = entry.path();
                if file_type.is_dir() {
                    pending.push(path);
                } else if file_type.is_file() && is_markdown(&path) {
                    pages.push(path);
                }
            }
        }
        // String order, not `Path` order: puts "Page <id>.md" before its "Page <id>/"
        // subpages, since '.' sorts before both separators.
        pages.sort_by_key(|p| p.to_string_lossy().into_owned());

        let mut docs = Vec::with_capacity(pages.len());
        for path in pages {
            let bytes = tokio::fs::read(&path).await?;
            let text = String::from_utf8(bytes)?;
            docs.push(Document::new(text).with_metadata(page_metadata(&path)));
        }
        Ok(docs)
    }
}

fn is_markdown(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("md"))
}

/// Notion names exported pages `<title> <32 hex chars>.md`.
fn split_notion_id(stem: &str) -> (&str, Option<&str>) {
    if let Some((title, id)) = stem.rsplit_once(' ') {
        if id.len() == 32 && id.bytes().all(|b| b.is_ascii_hexdigit()) && !title.is_empty() {
            return (title, Some(id));
        }
    }
    (stem, None)
}

fn page_metadata(path: &Path) -> HashMap<String, Value> {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let (title, id) = split_notion_id(&stem);
    let mut metadata = HashMap::from([
        ("source".to_string(), json!(path.to_string_lossy())),
        ("title".to_string(), json!(title)),
    ]);
    if let Some(id) = id {
        metadata.insert("notion_id".to_string(), json!(id));
    }
    metadata
}

#[async_trait]
impl Loader for NotionDirectoryLoader {
    async fn load(
        self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let docs = self.load_documents().await?;
        Ok(Box::pin(stream::iter(docs.into_iter().map(Ok))))
    }

    async fn load_and_split<TS: TextSplitter + 'static>(
        self,
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
    use super::*;
    use futures_util::StreamExt;

    const ID_A: &str = "1f2e3d4c5b6a79881f2e3d4c5b6a7988";
    const ID_B: &str = "abcdefabcdefabcdefabcdefabcdef01";

    /// A small export:
    /// ```text
    /// root/
    ///   Home <ID_A>.md
    ///   Home <ID_A>/
    ///     Roadmap <ID_B>.md
    ///     Tasks.csv
    ///     image.png
    ///     Deep/
    ///       notes.MD
    ///   README.txt
    /// ```
    fn export() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("lcr-notion-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let sub = root.join(format!("Home {ID_A}"));
        std::fs::create_dir_all(sub.join("Deep")).unwrap();
        std::fs::write(root.join(format!("Home {ID_A}.md")), "# Home\n\nWelcome.").unwrap();
        std::fs::write(sub.join(format!("Roadmap {ID_B}.md")), "# Roadmap\n\n- Q1").unwrap();
        std::fs::write(sub.join("Tasks.csv"), "Name,Status\nShip,Done").unwrap();
        std::fs::write(sub.join("image.png"), [0x89, b'P', b'N', b'G']).unwrap();
        std::fs::write(sub.join("Deep").join("notes.MD"), "deep notes").unwrap();
        std::fs::write(root.join("README.txt"), "not a page").unwrap();
        root
    }

    #[tokio::test]
    async fn loads_markdown_pages_recursively_in_path_order() {
        let root = export();
        let mut stream = NotionDirectoryLoader::new(&root).load().await.unwrap();
        let mut docs = Vec::new();
        while let Some(doc) = stream.next().await {
            docs.push(doc.unwrap());
        }
        let contents: Vec<&str> = docs.iter().map(|d| d.page_content.as_str()).collect();
        // Sorted by path: "Home <id>.md" < "Home <id>/..." ('.' < '/').
        assert_eq!(
            contents,
            ["# Home\n\nWelcome.", "deep notes", "# Roadmap\n\n- Q1"]
        );

        let home = &docs[0].metadata;
        assert_eq!(home["title"], json!("Home"));
        assert_eq!(home["notion_id"], json!(ID_A));
        assert_eq!(
            home["source"],
            json!(root.join(format!("Home {ID_A}.md")).to_string_lossy())
        );
        let deep = &docs[1].metadata;
        assert_eq!(deep["title"], json!("notes"));
        assert!(!deep.contains_key("notion_id"));
        assert_eq!(docs[2].metadata["notion_id"], json!(ID_B));
        assert_eq!(docs[2].metadata["title"], json!("Roadmap"));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn load_and_split_keeps_page_metadata() {
        use crate::text_splitter::{SplitterOptions, TokenSplitter};

        let root = export();
        let splitter = TokenSplitter::new(
            SplitterOptions::default()
                .with_chunk_size(2)
                .with_chunk_overlap(0),
        );
        let mut stream = NotionDirectoryLoader::new(&root)
            .load_and_split(splitter)
            .await
            .unwrap();
        let mut chunks = Vec::new();
        while let Some(doc) = stream.next().await {
            chunks.push(doc.unwrap());
        }
        assert!(chunks.len() > 3, "{} chunks", chunks.len());
        assert!(chunks.iter().all(|d| d.metadata.contains_key("title")));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn errors() {
        let missing = NotionDirectoryLoader::new("/definitely/not/a/notion/export")
            .load_documents()
            .await
            .unwrap_err();
        assert!(matches!(missing, LoaderError::IOError(_)), "{missing}");

        let root = export();
        std::fs::write(root.join("bad.md"), [0xff, 0xfe, 0x00]).unwrap();
        let bad = NotionDirectoryLoader::new(&root)
            .load_documents()
            .await
            .unwrap_err();
        assert!(matches!(bad, LoaderError::FromUtf8Error(_)), "{bad}");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn does_not_follow_symlinks() {
        let root = export();
        // A loop back to the root and a link to a page: neither is read.
        std::os::unix::fs::symlink(&root, root.join("loop")).unwrap();
        std::os::unix::fs::symlink(root.join(format!("Home {ID_A}.md")), root.join("link.md"))
            .unwrap();
        let docs = NotionDirectoryLoader::new(&root)
            .load_documents()
            .await
            .unwrap();
        assert_eq!(docs.len(), 3);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn notion_ids_are_split_from_titles() {
        assert_eq!(
            split_notion_id(&format!("My Page {ID_A}")),
            ("My Page", Some(ID_A))
        );
        assert_eq!(
            split_notion_id(&format!("A {}", ID_A.to_uppercase()))
                .1
                .map(str::len),
            Some(32)
        );
        for plain in [
            "My Page".to_string(),
            ID_A.to_string(),                 // id only, no title
            format!(" {ID_A}"),               // empty title
            format!("Page {}", &ID_A[..31]),  // 31 chars
            format!("Page {ID_A}0"),          // 33 chars
            format!("Page {}g", &ID_A[..31]), // not hex
            format!("Page_{ID_A}"),           // no space
        ] {
            assert_eq!(split_notion_id(&plain), (plain.as_str(), None), "{plain}");
        }
        assert!(is_markdown(Path::new("a/b.md")));
        assert!(is_markdown(Path::new("b.Md")));
        assert!(!is_markdown(Path::new("b.mdx")));
        assert!(!is_markdown(Path::new("md")));
        assert!(!is_markdown(Path::new("b.csv")));
    }
}
