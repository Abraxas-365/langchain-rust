use std::{error::Error, sync::Arc};

use tokio::sync::Mutex;

use crate::{
    chain::{llm_chain::LLMChainBuilder, options::ChainCallOptions, DEFAULT_OUTPUT_KEY},
    language_models::llm::LLM,
    memory::SimpleMemory,
    prompt::HumanMessagePromptTemplate,
    schemas::memory::BaseMemory,
    template_fstring,
};

use super::{prompt::DEFAULT_TEMPLATE, ConversationalChain};

pub struct ConversationalChainBuilder<L>
where
    L: LLM + 'static,
{
    llm: Option<L>,
    options: Option<ChainCallOptions>,
    memory: Option<Arc<Mutex<dyn BaseMemory>>>,
    output_key: Option<String>,
}

impl<L> ConversationalChainBuilder<L>
where
    L: LLM + 'static,
{
    pub fn new() -> Self {
        Self {
            llm: None,
            options: None,
            memory: None,
            output_key: None,
        }
    }

    pub fn llm(mut self, llm: L) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn options(mut self, options: ChainCallOptions) -> Self {
        self.options = Some(options);
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

    pub fn build(self) -> Result<ConversationalChain, Box<dyn Error>> {
        let llm = self.llm.ok_or("LLM must be set")?;
        let prompt = HumanMessagePromptTemplate::new(template_fstring!(
            DEFAULT_TEMPLATE,
            "history",
            "input"
        ));

        let llm_chain = match self.options {
            Some(options) => LLMChainBuilder::new()
                .prompt(prompt)
                .output_key(self.output_key.unwrap_or(DEFAULT_OUTPUT_KEY.into()))
                .llm(llm)
                .options(options)
                .build()?,
            None => LLMChainBuilder::new()
                .prompt(prompt)
                .llm(llm)
                .output_key(self.output_key.unwrap_or(DEFAULT_OUTPUT_KEY.into()))
                .build()?,
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
