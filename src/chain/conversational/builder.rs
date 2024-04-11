use std::sync::Arc;

use tokio::sync::Mutex;

use crate::{
    chain::{
        llm_chain::LLMChainBuilder, options::ChainCallOptions, ChainError, DEFAULT_OUTPUT_KEY,
    },
    language_models::llm::LLM,
    memory::SimpleMemory,
    output_parsers::OutputParser,
    prompt::HumanMessagePromptTemplate,
    schemas::memory::BaseMemory,
    template_fstring,
};

use super::{prompt::DEFAULT_TEMPLATE, ConversationalChain};

pub struct ConversationalChainBuilder {
    llm: Option<Box<dyn LLM>>,
    options: Option<ChainCallOptions>,
    memory: Option<Arc<Mutex<dyn BaseMemory>>>,
    output_key: Option<String>,
    output_parser: Option<Box<dyn OutputParser>>,
}

impl ConversationalChainBuilder {
    pub fn new() -> Self {
        Self {
            llm: None,
            options: None,
            memory: None,
            output_key: None,
            output_parser: None,
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

    pub fn output_parser<P: Into<Box<dyn OutputParser>>>(mut self, output_parser: P) -> Self {
        self.output_parser = Some(output_parser.into());
        self
    }

    pub fn memory(mut self, memory: Arc<Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn output_key<S: Into<String>>(mut self, output_key: S) -> Self {
        self.output_key = Some(output_key.into());
        self
    }

    pub fn build(self) -> Result<ConversationalChain, ChainError> {
        let llm = self
            .llm
            .ok_or_else(|| ChainError::MissingObject("LLM must be set".into()))?;
        let prompt = HumanMessagePromptTemplate::new(template_fstring!(
            DEFAULT_TEMPLATE,
            "history",
            "input"
        ));

        let llm_chain = {
            let mut builder = LLMChainBuilder::new()
                .prompt(prompt)
                .llm(llm)
                .output_key(self.output_key.unwrap_or_else(|| DEFAULT_OUTPUT_KEY.into()));

            if let Some(options) = self.options {
                builder = builder.options(options);
            }

            if let Some(output_parser) = self.output_parser {
                builder = builder.output_parser(output_parser);
            }

            builder.build()?
        };

        let memory = self
            .memory
            .unwrap_or_else(|| Arc::new(Mutex::new(SimpleMemory::new())));

        Ok(ConversationalChain {
            llm: llm_chain,
            memory,
        })
    }
}
