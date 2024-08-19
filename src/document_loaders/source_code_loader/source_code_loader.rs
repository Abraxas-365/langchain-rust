use crate::document_loaders::{
    find_files_with_extension, process_doc_stream, DirLoaderOptions, LoaderError,
};
use crate::{document_loaders::Loader, schemas::Document, text_splitter::TextSplitter};
use async_stream::stream;
use async_trait::async_trait;
use futures::Stream;

use std::fs::File;
use std::io::Read;
use std::pin::Pin;

use super::{get_language_by_filename, LanguageParser, LanguageParserOptions};

#[derive(Debug, Clone)]
pub struct SourceCodeLoader {
    file_path: Option<String>,
    string_input: Option<String>,
    dir_loader_options: DirLoaderOptions,
    parser_option: LanguageParserOptions,
}

impl SourceCodeLoader {
    pub fn from_string<S: Into<String>>(input: S) -> Self {
        Self {
            string_input: Some(input.into()),
            file_path: None,
            parser_option: LanguageParserOptions::default(),
            dir_loader_options: DirLoaderOptions::default(),
        }
    }
}

impl SourceCodeLoader {
    pub fn from_path<S: Into<String>>(path: S) -> Self {
        Self {
            file_path: Some(path.into()),
            string_input: None,
            parser_option: LanguageParserOptions::default(),
            dir_loader_options: DirLoaderOptions::default(),
        }
    }
}

impl SourceCodeLoader {
    pub fn with_parser_option(mut self, parser_option: LanguageParserOptions) -> Self {
        self.parser_option = parser_option;
        self
    }

    pub fn with_dir_loader_options(mut self, dir_loader_options: DirLoaderOptions) -> Self {
        self.dir_loader_options = dir_loader_options;
        self
    }
}

#[async_trait]
impl Loader for SourceCodeLoader {
    async fn load(
        mut self,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<Document, LoaderError>> + Send + 'static>>,
        LoaderError,
    > {
        let string_input = self.string_input.clone();
        let file_path = self.file_path.clone();

        if let Some(file_path) = file_path {
            let files =
                find_files_with_extension(file_path.as_str(), &self.dir_loader_options).await;
            let stream = stream! {
                for filename in files {
                    let mut file = match File::open(&filename) {
                        Ok(file) => file,
                        Err(e) => {
                            yield Err(LoaderError::OtherError(format!("Error opening file: {:?}", e)));
                            continue;
                        }
                    };
                    let mut content = String::new();
                    file.read_to_string(&mut content).unwrap();
                    let language = get_language_by_filename(&filename);
                    let mut parser = LanguageParser::from_language(language).with_parser_option(self.parser_option.clone());
                    let docs = parser.parse_code(&content);
                    for doc in docs {
                        yield Ok(doc);
                    }
                }
            };

            return Ok(Box::pin(stream));
        } else if let Some(content) = string_input {
            let language = self.parser_option.language.clone();
            let stream = stream! {
                    let mut parser = LanguageParser::from_language(language).with_parser_option(self.parser_option.clone());
                    let docs = parser.parse_code(&content);
                    for doc in docs {
                        yield Ok(doc);
                    }
            };

            return Ok(Box::pin(stream));
        }
        Err(LoaderError::OtherError(
            "No file path or string input provided".to_string(),
        ))
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

    use crate::document_loaders::{Language, LanguageContentTypes};

    use super::*;

    #[tokio::test]
    async fn test_sourse_code_loader() {
        let loader_with_dir =
            SourceCodeLoader::from_path("./src/document_loaders/test_data".to_string())
                .with_dir_loader_options(DirLoaderOptions {
                    glob: None,
                    suffixes: Some(vec!["rs".to_string()]),
                    exclude_dirs: None,
                    exclude_files: None
                });

        let stream = loader_with_dir.load().await.unwrap();
        let documents = stream.map(|x| x.unwrap()).collect::<Vec<_>>().await;

        assert_eq!(documents.len(), 1);
        assert_eq!(
            documents[0].metadata.get("content_type").unwrap(),
            LanguageContentTypes::SimplifiedCode.to_string().as_str()
        );

        let loader_with_file =
            SourceCodeLoader::from_path("./src/document_loaders/test_data/example.rs".to_string());
        let documents_with_file = loader_with_file
            .load()
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await;
        assert_eq!(documents_with_file.len(), documents.len());

        let mut example_file = File::open("./src/document_loaders/test_data/example.rs").unwrap();
        let mut example_content = String::new();
        example_file.read_to_string(&mut example_content).unwrap();
        let loader_with_content = SourceCodeLoader::from_string(example_content)
            .with_parser_option(LanguageParserOptions {
                language: Language::Rust,
                ..Default::default()
            });
        let documents_with_content = loader_with_content
            .load()
            .await
            .unwrap()
            .map(|x| x.unwrap())
            .collect::<Vec<_>>()
            .await;
        assert_eq!(documents_with_content.len(), documents.len());
    }
}
