use std::sync::Arc;

use crate::{
    agent::AgentError,
    chain::{llm_chain::LLMChainBuilder, options::ChainCallOptions},
    language_models::llm::LLM,
    prompt::FormatPrompter,
    tools::Tool,
};

use super::{
    prompt::{DEFAULT_INITIAL_PROMPT, DEFAULT_SYSTEM_PROMPT},
    ConversationalAgent,
};

pub struct ConversationalAgentBuilder<'a, 'b> {
    tools: Option<Vec<Arc<dyn Tool>>>,
    system_prompt: Option<&'a str>,
    initial_prompt: Option<&'b str>,
    options: Option<ChainCallOptions>,
}

impl<'a, 'b> ConversationalAgentBuilder<'a, 'b> {
    pub fn new() -> Self {
        Self {
            tools: None,
            system_prompt: None,
            initial_prompt: None,
            options: None,
        }
    }

    pub fn tools(mut self, tools: &[Arc<dyn Tool>]) -> Self {
        self.tools = Some(tools.to_vec());
        self
    }

    pub fn system_prompt<S: Into<String>>(mut self, system_prompt: &'a str) -> Self {
        self.system_prompt = Some(system_prompt);
        self
    }

    pub fn initial_prompt<S: Into<String>>(mut self, initial_prompt: &'b str) -> Self {
        self.initial_prompt = Some(initial_prompt);
        self
    }

    pub fn options(mut self, options: ChainCallOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn build<L: Into<Box<dyn LLM>>>(self, llm: L) -> Result<ConversationalAgent, AgentError> {
        let tools = self.tools.unwrap_or_default();
        let system_prompt = self.system_prompt.unwrap_or(DEFAULT_SYSTEM_PROMPT);
        let initial_prompt = self.initial_prompt.unwrap_or(DEFAULT_INITIAL_PROMPT);

        let prompt = ConversationalAgent::create_prompt(system_prompt, initial_prompt, &tools)?;
        let default_options = ChainCallOptions::default().with_max_tokens(1000);
        let chain = Box::new(
            LLMChainBuilder::new()
                .prompt(Box::new(prompt) as Box<dyn FormatPrompter<_>>)
                .llm(llm)
                .options(self.options.unwrap_or(default_options))
                .build()?,
        );

        Ok(ConversationalAgent { chain, tools })
    }
}

impl Default for ConversationalAgentBuilder<'_, '_> {
    fn default() -> Self {
        Self::new()
    }
}
