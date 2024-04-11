use crate::{
    chain::{
        llm_chain::LLMChainBuilder, options::ChainCallOptions, ChainError, DEFAULT_OUTPUT_KEY,
    },
    language_models::llm::LLM,
    output_parsers::OutputParser,
    prompt::HumanMessagePromptTemplate,
    template_jinja2,
    tools::SQLDatabase,
};

use super::{
    chain::SQLDatabaseChain,
    prompt::{DEFAULT_SQLSUFFIX, DEFAULT_SQLTEMPLATE},
    STOP_WORD,
};

pub struct SQLDatabaseChainBuilder {
    llm: Option<Box<dyn LLM>>,
    options: Option<ChainCallOptions>,
    top_k: Option<usize>,
    database: Option<SQLDatabase>,
    output_key: Option<String>,
    output_parser: Option<Box<dyn OutputParser>>,
}

impl SQLDatabaseChainBuilder {
    pub fn new() -> Self {
        Self {
            llm: None,
            options: None,
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

    pub fn options(mut self, options: ChainCallOptions) -> Self {
        self.options = Some(options);
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

        let prompt = HumanMessagePromptTemplate::new(template_jinja2!(
            format!("{}{}", DEFAULT_SQLTEMPLATE, DEFAULT_SQLSUFFIX),
            "dialect",
            "table_info",
            "top_k",
            "input"
        ));

        let llm_chain = {
            let mut builder = LLMChainBuilder::new()
                .prompt(prompt)
                .output_key(self.output_key.unwrap_or_else(|| DEFAULT_OUTPUT_KEY.into()))
                .llm(llm);

            let mut options = self.options.unwrap_or_else(ChainCallOptions::default);
            options = options.with_stop_words(vec![STOP_WORD.to_string()]);
            builder = builder.options(options);

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
