use std::{collections::HashMap, sync::Arc};

use crate::{
    agent::AgentError,
    chain::{options::ChainCallOptions, LLMChainBuilder},
    language_models::{llm::LLM, options::CallOptions},
    schemas::FunctionDefinition,
    tools::Tool,
};

use super::{prompt::PREFIX, OpenAiToolAgent};

pub struct OpenAiToolAgentBuilder {
    tools: Option<HashMap<String, Arc<dyn Tool>>>,
    prefix: Option<String>,
    options: Option<ChainCallOptions>,
}

impl OpenAiToolAgentBuilder {
    pub fn new() -> Self {
        Self {
            tools: None,
            prefix: None,
            options: None,
        }
    }

    pub fn tools(mut self, tools: HashMap<String, Arc<dyn Tool>>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.prefix = Some(prefix.into());
        self
    }

    pub fn options(mut self, options: ChainCallOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn build<L: LLM + 'static>(self, llm: L) -> Result<OpenAiToolAgent, AgentError> {
        let tools = self.tools.unwrap_or_default();
        let prefix = self.prefix.unwrap_or_else(|| PREFIX.to_string());
        let mut llm = llm;

        let prompt = OpenAiToolAgent::create_prompt(&prefix)?;
        let default_options = ChainCallOptions::default().with_max_tokens(1000);
        let functions = tools
            .values()
            .map(FunctionDefinition::from_langchain_tool)
            .collect::<Vec<FunctionDefinition>>();
        llm.add_options(CallOptions::new().with_functions(functions));
        let chain = Box::new(
            LLMChainBuilder::new()
                .prompt(prompt)
                .llm(llm)
                .options(self.options.unwrap_or(default_options))
                .build()?,
        );

        Ok(OpenAiToolAgent { chain, tools })
    }
}

impl Default for OpenAiToolAgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
