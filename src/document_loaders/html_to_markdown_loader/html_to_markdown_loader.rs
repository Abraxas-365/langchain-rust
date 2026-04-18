use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Cursor, Read},
    path::Path,
    pin::Pin,
};

use async_trait::async_trait;
use futures::{stream, Stream};
use serde_json::Value;
use url::Url;

pub use htmd::{HtmlToMarkdown, HtmlToMarkdownBuilder};

use crate::{
    document_loaders::{process_doc_stream, Loader, LoaderError},
    schemas::Document,
    text_splitter::TextSplitter,
};

#[derive(Debug, Clone)]
pub struct HtmlToMarkdownLoader<R> {
    html: R,
    url: Url,
    options: HtmlToMarkdownLoaderOptions,
}

#[derive(Debug, Clone, Default)]
pub struct HtmlToMarkdownLoaderOptions {
    skip_tags: Option<Vec<String>>,
}

impl HtmlToMarkdownLoaderOptions {
    pub fn with_skip_tags(mut self, tags: Vec<String>) -> Self {
        self.skip_tags = Some(tags);
        self
    }

    pub fn skip_tags(&self) -> Option<&Vec<String>> {
        self.skip_tags.as_ref()
    }
}

impl HtmlToMarkdownLoader<Cursor<Vec<u8>>> {
    pub fn from_string<S: Into<String>>(
        input: S,
        url: Url,
        options: HtmlToMarkdownLoaderOptions,
    ) -> Self {
        let input = input.into();
        let reader = Cursor::new(input.into_bytes());
        Self::new(reader, url, options)
    }
}

impl<R: Read> HtmlToMarkdownLoader<R> {
    pub fn new(html: R, url: Url, options: HtmlToMarkdownLoaderOptions) -> Self {
        Self { html, url, options }
    }
}

impl HtmlToMarkdownLoader<BufReader<File>> {
    pub fn from_path<P: AsRef<Path>>(
        path: P,
        url: Url,
        options: HtmlToMarkdownLoaderOptions,
    ) -> Result<Self, LoaderError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(Self::new(reader, url, options))
    }
}

#[async_trait]
impl<R: Read + Send + Sync + 'static> Loader for HtmlToMarkdownLoader<R> {
    async fn load(
        mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let mut converter_builder = HtmlToMarkdownBuilder::default();
        if let Some(skip_tags) = &self.options.skip_tags {
            converter_builder =
                converter_builder.skip_tags(skip_tags.iter().map(|s| s.as_str()).collect());
        }
        let converter = converter_builder.build();

        let mut buffer = String::new();
        self.html.read_to_string(&mut buffer)?;
        let cleand_html = converter.convert(&buffer)?;

        let doc = Document::new(cleand_html).with_metadata(HashMap::from([(
            "source".to_string(),
            Value::from(self.url.as_str()),
        )]));

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
    async fn test_html_to_markdown_loader() {
        let input = "<h1>Page Title</h1><h2>Sub Title</h2><p>Hello world!</p>";

        let html_loader = HtmlToMarkdownLoader::new(
            input.as_bytes(),
            Url::parse("https://example.com/").unwrap(),
            HtmlToMarkdownLoaderOptions::default(),
        );

        let documents = html_loader
            .load()
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await;

        let expected = "# Page Title\n\n## Sub Title\n\nHello world!";

        assert_eq!(documents.len(), 1);
        assert_eq!(
            documents[0].metadata.get("source").unwrap(),
            &Value::from("https://example.com/")
        );
        assert_eq!(documents[0].page_content, expected);
    }

    #[tokio::test]
    async fn test_html_to_markdown_loader_with_skip_tags() {
        let input = "<h1>Page Title</h1><h2>Sub Title</h2><p>Hello world!</p>";

        let html_loader = HtmlToMarkdownLoader::new(
            input.as_bytes(),
            Url::parse("https://example.com/").unwrap(),
            HtmlToMarkdownLoaderOptions::default().with_skip_tags(vec!["h2".to_string()]),
        );

        let documents = html_loader
            .load()
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await;

        let expected = "# Page Title\n\nHello world!";

        assert_eq!(documents.len(), 1);
        assert_eq!(
            documents[0].metadata.get("source").unwrap(),
            &Value::from("https://example.com/")
        );
        assert_eq!(documents[0].page_content, expected);
    }

    #[tokio::test]
    async fn test_html_to_markdown_load_from_path() {
        let path = "./src/document_loaders/test_data/example.html";
        let html_loader = HtmlToMarkdownLoader::from_path(
            path,
            Url::parse("https://example.com/").unwrap(),
            HtmlToMarkdownLoaderOptions::default(),
        )
        .expect("Failed to create html loader");

        let documents = html_loader
            .load()
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await;

        let expected = "Chew dad's slippers\n\n# Instead of drinking water from the cat bowl, make sure to steal water from the toilet\n\n## Chase the red dot\n\nMunch, munch, chomp, chomp hate dogs. Spill litter box, scratch at owner, destroy all furniture, especially couch get scared by sudden appearance of cucumber cat is love, cat is life fat baby cat best buddy little guy for catch eat throw up catch eat throw up bad birds jump on fridge. Purr like a car engine oh yes, there is my human woman she does best pats ever that all i like about her hiss meow . \n\nDead stare with ears cocked when owners are asleep, cry for no apparent reason meow all night. Plop down in the middle where everybody walks favor packaging over toy. Sit on the laptop kitty pounce, trip, faceplant.";

        assert_eq!(documents.len(), 1);
        assert_eq!(
            documents[0].metadata.get("source").unwrap(),
            &Value::from("https://example.com/")
        );
        assert_eq!(documents[0].page_content, expected);
    }
}
