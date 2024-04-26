use crate::{
    chain::{options::ChainCallOptions, ChainError, LLMChainBuilder},
    language_models::llm::LLM,
    output_parsers::OutputParser,
    prompt::FormatPrompter,
    template_jinja2,
};

use super::StuffDocument;

pub struct StuffDocumentBuilder {
    llm: Option<Box<dyn LLM>>,
    options: Option<ChainCallOptions>,
    output_key: Option<String>,
    output_parser: Option<Box<dyn OutputParser>>,
    prompt: Option<Box<dyn FormatPrompter>>,
}
impl StuffDocumentBuilder {
    pub fn new() -> Self {
        Self {
            llm: None,
            options: None,
            output_key: None,
            output_parser: None,
            prompt: None,
        }
    }

    pub fn llm<L: Into<Box<dyn LLM>>>(mut self, llm: L) -> Self {
        self.llm = Some(llm.into());
        self
    }

    pub fn options(mut self, options: ChainCallOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn output_key<S: Into<String>>(mut self, output_key: S) -> Self {
        self.output_key = Some(output_key.into());
        self
    }

    pub fn prompt<P: Into<Box<dyn FormatPrompter>>>(mut self, prompt: P) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn build(self) -> Result<StuffDocument, ChainError> {
        let llm = self
            .llm
            .ok_or_else(|| ChainError::MissingObject("LLM must be set".into()))?;
        let prompt = match self.prompt {
            Some(prompt) => prompt,
            None => Box::new(template_jinja2!(
                DEFAULT_STUFF_QA_TEMPLATE,
                "context",
                "question"
            )),
        };

        let llm_chain = {
            let mut builder = LLMChainBuilder::new()
                .prompt(prompt)
                .options(self.options.unwrap_or_default())
                .llm(llm);
            if let Some(output_parser) = self.output_parser {
                builder = builder.output_parser(output_parser);
            }

            builder.build()?
        };

        Ok(StuffDocument::new(llm_chain))
    }
}

const DEFAULT_STUFF_QA_TEMPLATE: &str = r#"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{{context}}

Question:{{question}}
Helpful Answer:
"#;
