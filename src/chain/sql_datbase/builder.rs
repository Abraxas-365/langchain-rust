use crate::{
    chain::{llm_chain::LLMChainBuilder, ChainError},
    language_models::{llm::LLM, options::CallOptions},
    output_parsers::OutputParser,
    prompt_template,
    schemas::MessageType,
    template::{MessageTemplate, PromptTemplate},
    tools::SQLDatabase,
};

use super::{
    chain::SQLDatabaseChain,
    prompt::{DEFAULT_SQLSUFFIX, DEFAULT_SQLTEMPLATE},
    STOP_WORD,
};

pub struct SQLDatabaseChainBuilder {
    llm: Option<Box<dyn LLM>>,
    top_k: Option<usize>,
    database: Option<SQLDatabase>,
    output_key: Option<String>,
    output_parser: Option<Box<dyn OutputParser>>,
}

impl SQLDatabaseChainBuilder {
    pub fn new() -> Self {
        Self {
            llm: None,
            top_k: None,
            database: None,
            output_key: None,
            output_parser: None,
        }
    }

    pub fn llm<L: Into<Box<dyn LLM>>>(mut self, llm: L) -> Self {
        self.llm = Some(llm.into());
        self
    }

    pub fn output_key<S: Into<String>>(mut self, output_key: S) -> Self {
        self.output_key = Some(output_key.into());
        self
    }

    pub fn output_parser<P: Into<Box<dyn OutputParser>>>(mut self, output_parser: P) -> Self {
        self.output_parser = Some(output_parser.into());
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn database(mut self, database: SQLDatabase) -> Self {
        self.database = Some(database);
        self
    }

    pub fn build(self) -> Result<SQLDatabaseChain, ChainError> {
        let llm = self
            .llm
            .ok_or_else(|| ChainError::MissingObject("LLM must be set".into()))?;
        let top_k = self
            .top_k
            .ok_or_else(|| ChainError::MissingObject("Top K must be set".into()))?;
        let database = self
            .database
            .ok_or_else(|| ChainError::MissingObject("Database must be set".into()))?;

        let prompt: PromptTemplate = prompt_template![MessageTemplate::from_jinja2(
            MessageType::HumanMessage,
            &format!("{}{}", DEFAULT_SQLTEMPLATE, DEFAULT_SQLSUFFIX),
        )];

        let llm_chain = {
            let mut llm = llm.clone_box();
            llm.add_options(CallOptions::new().with_stop_words(vec![STOP_WORD.to_string()]));

            let mut builder = LLMChainBuilder::new().prompt(prompt).llm(llm);

            if let Some(output_parser) = self.output_parser {
                builder = builder.output_parser(output_parser);
            }

            builder.build()?
        };

        Ok(SQLDatabaseChain {
            llmchain: llm_chain,
            top_k,
            database,
        })
    }
}

impl Default for SQLDatabaseChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}
