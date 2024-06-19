use std::{fmt, path::Path, pin::Pin, process::Stdio};

use async_trait::async_trait;
use futures_util::{stream, Stream};
use tokio::{
    fs::File,
    io::{AsyncRead, AsyncWriteExt, BufReader},
    process::Command,
};

use crate::{
    document_loaders::{process_doc_stream, Loader, LoaderError},
    schemas::Document,
    text_splitter::TextSplitter,
};

#[derive(Debug)]
pub enum InputFormat {
    Docx,
    Epub,
    Html,
    JuypterNotebook,
    Markdown,
    MediaWiki,
    RichTextFormat,
    Typst,
    VimWiki,
}

impl ToString for InputFormat {
    fn to_string(&self) -> String {
        match self {
            InputFormat::Docx => "docx".into(),
            InputFormat::Epub => "epub".into(),
            InputFormat::Html => "html".into(),
            InputFormat::JuypterNotebook => "ipynb".into(),
            InputFormat::MediaWiki => "mediawiki".into(),
            InputFormat::Markdown => "markdown".into(),
            InputFormat::RichTextFormat => "rtf".into(),
            InputFormat::Typst => "typst".into(),
            InputFormat::VimWiki => "vimwiki".into(),
        }
    }
}

pub struct PandocLoader<R> {
    pandoc_path: String,
    input_format: String,
    input: R,
}

impl<R> fmt::Debug for PandocLoader<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PandocLoader")
            .field("pandoc_path", &self.pandoc_path)
            .field("input_format", &self.input_format)
            .finish()
    }
}

impl<R: AsyncRead + Send + Sync + Unpin + 'static> PandocLoader<R> {
    pub fn new<S: Into<String>>(pandoc_path: S, input_format: S, input: R) -> Self {
        PandocLoader {
            pandoc_path: pandoc_path.into(),
            input_format: input_format.into(),
            input,
        }
    }

    pub fn new_from_reader<S: Into<String>>(input_format: S, input: R) -> Self {
        PandocLoader::new("pandoc".into(), input_format.into(), input)
    }

    pub fn with_pandoc_path<S: Into<String>>(mut self, pandoc_path: S) -> Self {
        self.pandoc_path = pandoc_path.into();
        self
    }
}

impl PandocLoader<BufReader<File>> {
    pub async fn from_path<P: AsRef<Path>, S: Into<String>>(
        input_format: S,
        path: P,
    ) -> Result<Self, LoaderError> {
        let file = File::open(path).await?;
        let reader = BufReader::new(file);

        Ok(Self::new("pandoc".into(), input_format.into(), reader))
    }
}

#[async_trait]
impl<R: AsyncRead + Send + Sync + Unpin + 'static> Loader for PandocLoader<R> {
    async fn load(
        mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        // echo "# Heading1 \n ## Heading 2 \n this is a markdown" | pandoc -f markdown -t plain
        // cat test.md | pandoc -f markdown -t plain

        let mut process = Command::new(self.pandoc_path)
            .arg("-f")
            .arg(self.input_format)
            .arg("-t")
            .arg("plain")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        // safe to unwrap since stdout/stdout has been configured.
        let mut stdin = process.stdin.take().unwrap();
        let mut stdout = process.stdout.take().unwrap();

        tokio::spawn(async move {
            match tokio::io::copy(&mut self.input, &mut stdin).await {
                Ok(_) => {}
                Err(e) => {
                    log::error!("pandoc stdin error: {}", e.to_string());
                }
            }
            stdin.flush().await.unwrap();
            stdin.shutdown().await.unwrap();
        });

        let stdout_task = tokio::spawn(async move {
            let mut buffer = Vec::new();
            match tokio::io::copy(&mut stdout, &mut buffer).await {
                Ok(_) => Ok(buffer),
                Err(e) => Err(e),
            }
        });

        let _exit_status = process.wait().await?;
        let stdout_result = stdout_task.await?.unwrap();
        let stdout_string = String::from_utf8(stdout_result).map_err(|e| {
            LoaderError::OtherError(format!("Failed to convert to utf8 string: {}", e))
        })?;

        let doc = Document::new(stdout_string);
        let stream = stream::iter(vec![Ok(doc)]);
        Ok(Box::pin(stream))
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
    async fn test_pandoc_loader() {
        let path = "./src/document_loaders/test_data/sample.docx";

        let loader = PandocLoader::from_path(InputFormat::Docx.to_string(), path)
            .await
            .expect("Failed to create PandocLoader");

        let docs = loader
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;

        // only pick the first 27 characters for now
        assert_eq!(&docs[0].page_content[..27], "Lorem ipsum dolor sit amet,");
        assert_eq!(docs.len(), 1);
    }
}
