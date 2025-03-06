use std::sync::Arc;

use crate::{
    agent::AgentError,
    chain::{llm_chain::LLMChainBuilder, options::ChainCallOptions},
    language_models::llm::LLM,
    tools::Tool,
};

use super::{
    output_parser::ChatOutputParser,
    prompt::{PREFIX, SUFFIX},
    ConversationalAgent,
};

pub struct ConversationalAgentBuilder {
    tools: Option<Vec<Arc<dyn Tool>>>,
    prefix: Option<String>,
    suffix: Option<String>,
    options: Option<ChainCallOptions>,
}

impl ConversationalAgentBuilder {
    pub fn new() -> Self {
        Self {
            tools: None,
            prefix: None,
            suffix: None,
            options: None,
        }
    }

    pub fn tools(mut self, tools: &[Arc<dyn Tool>]) -> Self {
        self.tools = Some(tools.to_vec());
        self
    }

    pub fn prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    pub fn suffix<S: Into<String>>(mut self, suffix: S) -> Self {
        self.suffix = Some(suffix.into());
        self
    }

    pub fn options(mut self, options: ChainCallOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn build<L: Into<Box<dyn LLM>>>(self, llm: L) -> Result<ConversationalAgent, AgentError> {
        let tools = self.tools.unwrap_or_default();
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
            output_parser: ChatOutputParser::new(),
        })
    }
}

impl Default for ConversationalAgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
