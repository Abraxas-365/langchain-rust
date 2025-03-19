use std::sync::Arc;

use tokio::sync::Mutex;

use crate::{
    chain::{
        llm_chain::LLMChainBuilder, options::ChainCallOptions, ChainError, DEFAULT_OUTPUT_KEY,
    },
    language_models::llm::LLM,
    memory::SimpleMemory,
    output_parsers::OutputParser,
    schemas::{memory::BaseMemory, MessageTemplate, MessageType, PromptTemplate},
};

use super::{prompt::DEFAULT_TEMPLATE, ConversationalChain, DEFAULT_INPUT_VARIABLE};

pub struct ConversationalChainBuilder {
    llm: Option<Box<dyn LLM>>,
    options: Option<ChainCallOptions>,
    memory: Option<Arc<Mutex<dyn BaseMemory>>>,
    output_key: Option<String>,
    output_parser: Option<Box<dyn OutputParser>>,
    input_key: Option<String>,
    prompt: Option<PromptTemplate>,
}

impl ConversationalChainBuilder {
    pub fn new() -> Self {
        Self {
            llm: None,
            options: None,
            memory: None,
            output_key: None,
            output_parser: None,
            input_key: None,
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

    pub fn input_key<S: Into<String>>(mut self, input_key: S) -> Self {
        self.input_key = Some(input_key.into());
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

    ///If you want to add a custom prompt,keep in mind which variables are obligatory.
    pub fn prompt<P: Into<PromptTemplate>>(mut self, prompt: P) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn build(self) -> Result<ConversationalChain, ChainError> {
        let llm = self
            .llm
            .ok_or_else(|| ChainError::MissingObject("LLM must be set".into()))?;
        let prompt = match self.prompt {
            Some(prompt) => prompt,
            None => {
                MessageTemplate::from_fstring(MessageType::HumanMessage, DEFAULT_TEMPLATE).into()
            }
        };
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
            input_key: self
                .input_key
                .unwrap_or_else(|| DEFAULT_INPUT_VARIABLE.to_string()),
        })
    }
}

impl Default for ConversationalChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}
