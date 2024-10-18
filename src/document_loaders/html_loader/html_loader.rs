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

use crate::{
    document_loaders::{process_doc_stream, Loader, LoaderError},
    schemas::Document,
    text_splitter::TextSplitter,
};

#[derive(Debug, Clone)]
pub struct HtmlLoader<R> {
    html: R,
    url: Url,
}

impl HtmlLoader<Cursor<Vec<u8>>> {
    pub fn from_string<S: Into<String>>(input: S, url: Url) -> Self {
        let input = input.into();
        let reader = Cursor::new(input.into_bytes());
        Self::new(reader, url)
    }
}

impl<R: Read> HtmlLoader<R> {
    pub fn new(html: R, url: Url) -> Self {
        Self { html, url }
    }
}

impl HtmlLoader<BufReader<File>> {
    pub fn from_path<P: AsRef<Path>>(path: P, url: Url) -> Result<Self, LoaderError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(Self::new(reader, url))
    }
}

#[async_trait]
impl<R: Read + Send + Sync + 'static> Loader for HtmlLoader<R> {
    async fn load(
        mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let cleaned_html = readability::extractor::extract(&mut self.html, &self.url)?;
        let doc =
            Document::new(format!("{}\n{}", cleaned_html.title, cleaned_html.text)).with_metadata(
                HashMap::from([("source".to_string(), Value::from(self.url.as_str()))]),
            );

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
    async fn test_html_loader() {
        let input = "<p>Hello world!</p>";

        let html_loader = HtmlLoader::new(
            input.as_bytes(),
            Url::parse("https://example.com/").unwrap(),
        );

        let documents = html_loader
            .load()
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await;

        let expected = "\nHello world!";

        assert_eq!(documents.len(), 1);
        assert_eq!(
            documents[0].metadata.get("source").unwrap(),
            &Value::from("https://example.com/")
        );
        assert_eq!(documents[0].page_content, expected);
    }

    #[tokio::test]
    async fn test_html_load_from_path() {
        let path = "./src/document_loaders/test_data/example.html";
        let html_loader = HtmlLoader::from_path(path, Url::parse("https://example.com/").unwrap())
            .expect("Failed to create html loader");

        let documents = html_loader
            .load()
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await;

        let expected = "Chew dad's slippers\nChase the red dot\n      Munch, munch, chomp, chomp hate dogs. Spill litter box, scratch at owner,\n      destroy all furniture, especially couch get scared by sudden appearance of\n      cucumber cat is love, cat is life fat baby cat best buddy little guy for\n      catch eat throw up catch eat throw up bad birds jump on fridge. Purr like\n      a car engine oh yes, there is my human woman she does best pats ever that\n      all i like about her hiss meow .\n    \n      Dead stare with ears cocked when owners are asleep, cry for no apparent\n      reason meow all night. Plop down in the middle where everybody walks favor\n      packaging over toy. Sit on the laptop kitty pounce, trip, faceplant.\n    ";

        assert_eq!(documents.len(), 1);
        assert_eq!(
            documents[0].metadata.get("source").unwrap(),
            &Value::from("https://example.com/")
        );
        assert_eq!(documents[0].page_content, expected);
    }
}
