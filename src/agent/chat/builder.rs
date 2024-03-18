use std::{error::Error, sync::Arc};

use crate::{
    agent::agent::AgentOutputParser,
    chain::{llm_chain::LLMChainBuilder, options::ChainCallOptions},
    language_models::llm::LLM,
    tools::Tool,
};

use super::{
    prompt::{PREFIX, SUFFIX},
    ConversationalAgent,
};

pub struct ConversationalAgentBuilder {
    tools: Option<Vec<Arc<dyn Tool>>>,
    output_parser: Option<Box<dyn AgentOutputParser>>,
    prefix: Option<String>,
    suffix: Option<String>,
    options: Option<ChainCallOptions>,
}

impl ConversationalAgentBuilder {
    pub fn new() -> Self {
        Self {
            tools: None,
            output_parser: None,
            prefix: None,
            suffix: None,
            options: None,
        }
    }

    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn output_parser(mut self, output_parser: Box<dyn AgentOutputParser>) -> Self {
        self.output_parser = Some(output_parser);
        self
    }

    pub fn prefix(mut self, prefix: String) -> Self {
        self.prefix = Some(prefix);
        self
    }

    pub fn suffix(mut self, suffix: String) -> Self {
        self.suffix = Some(suffix);
        self
    }

    pub fn options(mut self, options: ChainCallOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn build<L: LLM + 'static>(self, llm: L) -> Result<ConversationalAgent, Box<dyn Error>> {
        let tools = self.tools.unwrap_or_else(Vec::new);
        let output_parser = self.output_parser.ok_or("Output parser must be set")?;
        let prefix = self.prefix.unwrap_or_else(|| PREFIX.to_string());
        let suffix = self.suffix.unwrap_or_else(|| SUFFIX.to_string());

        let prompt = ConversationalAgent::create_prompt(&tools, &suffix, &prefix)?;
        let default_options = ChainCallOptions::default().with_max_tokens(1000);
        let chain = Box::new(
            LLMChainBuilder::new()
                .prompt(prompt)
                .llm(llm)
                .options(self.options.unwrap_or(default_options))
                .build()?,
        );

        Ok(ConversationalAgent {
            chain,
            tools,
            output_parser,
        })
    }
}
