use std::{collections::HashMap, sync::Arc};

use crate::{
    agent::AgentError,
    chain::{llm_chain::LLMChainBuilder, options::ChainCallOptions},
    language_models::llm::LLM,
    tools::Tool,
};

use super::{
    agent::OBSERVATION_STOP_WORDS,
    output_parser::ReActOutputParser,
    prompt::{ReActPrompt, FORMAT_INSTRUCTIONS, PREFIX, SUFFIX},
    ReActAgent,
};

/// Builds a [`ReActAgent`].
///
/// ```rust,ignore
/// let agent = ReActAgentBuilder::new()
///     .tools(&[Arc::new(calculator)])
///     .build(llm)?;
/// let executor = AgentExecutor::from_agent(agent)
///     .with_parser_error_handler(ParserErrorHandler::new());
/// let answer = executor.invoke(prompt_args! { "input" => "What is 3 to the 5th?" }).await?;
/// ```
pub struct ReActAgentBuilder {
    tools: Vec<Arc<dyn Tool>>,
    prefix: String,
    format_instructions: String,
    suffix: String,
    options: Option<ChainCallOptions>,
    stop_at_observation: bool,
}

impl Default for ReActAgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ReActAgentBuilder {
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            prefix: PREFIX.to_string(),
            format_instructions: FORMAT_INSTRUCTIONS.to_string(),
            suffix: SUFFIX.to_string(),
            options: None,
            stop_at_observation: true,
        }
    }

    pub fn tools(mut self, tools: &[Arc<dyn Tool>]) -> Self {
        self.tools = tools.to_vec();
        self
    }

    /// Replaces the text before the format instructions. `{{tool_descriptions}}` and
    /// `{{tool_names}}` are filled in.
    pub fn prefix<S: Into<String>>(mut self, prefix: S) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Replaces the format instructions. The output parser still expects `Action:` /
    /// `Action Input:` or `Final Answer:`.
    pub fn format_instructions<S: Into<String>>(mut self, format_instructions: S) -> Self {
        self.format_instructions = format_instructions.into();
        self
    }

    /// Replaces the text after the format instructions. It must contain `{{input}}` and
    /// `{{agent_scratchpad}}`; `{{chat_history}}` is optional.
    pub fn suffix<S: Into<String>>(mut self, suffix: S) -> Self {
        self.suffix = suffix.into();
        self
    }

    /// Model options. The `Observation:` stop words are added to them (see
    /// [`stop_at_observation`](Self::stop_at_observation)).
    pub fn options(mut self, options: ChainCallOptions) -> Self {
        self.options = Some(options);
        self
    }

    /// Whether to stop generation at `Observation:` (default `true`), so the model doesn't
    /// make up the tool's answer. Turn it off for models that reject stop sequences (e.g.
    /// OpenAI reasoning models); the parser then ignores anything after `Observation`.
    pub fn stop_at_observation(mut self, stop: bool) -> Self {
        self.stop_at_observation = stop;
        self
    }

    /// The prompt, with the tools to fill in.
    fn prompt(&self) -> Result<ReActPrompt, AgentError> {
        for var in ["{{input}}", "{{agent_scratchpad}}"] {
            if !self.suffix.contains(var) && !self.prefix.contains(var) {
                return Err(AgentError::MissingInputVariable(format!(
                    "the ReAct prompt needs {var}"
                )));
            }
        }
        let tool_descriptions = self
            .tools
            .iter()
            .map(|tool| format!("{}: {}", tool.name(), tool.description()))
            .collect::<Vec<_>>()
            .join("\n");
        let tool_names = self
            .tools
            .iter()
            .map(|tool| tool.name())
            .collect::<Vec<_>>()
            .join(", ");
        let template = [&self.prefix, &self.format_instructions, &self.suffix].map(String::as_str);
        Ok(ReActPrompt::new(
            template.join("\n\n"),
            HashMap::from([
                ("tool_descriptions", tool_descriptions),
                ("tool_names", tool_names),
            ]),
        ))
    }

    pub fn build<L: Into<Box<dyn LLM>>>(self, llm: L) -> Result<ReActAgent, AgentError> {
        let prompt = self.prompt()?;

        let mut options = self.options.unwrap_or_default();
        if self.stop_at_observation {
            let stop_words = options.stop_words.get_or_insert_with(Vec::new);
            for word in OBSERVATION_STOP_WORDS {
                if !stop_words.iter().any(|w| w == word) {
                    stop_words.push(word.to_string());
                }
            }
        }
        let chain = LLMChainBuilder::new()
            .prompt(prompt)
            .llm(llm)
            .options(options)
            .build()?;

        Ok(ReActAgent {
            chain: Box::new(chain),
            tools: self.tools,
            output_parser: ReActOutputParser::new(),
        })
    }
}
