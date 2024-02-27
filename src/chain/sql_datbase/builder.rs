use std::error::Error;

use crate::{
    chain::{llm_chain::LLMChainBuilder, options::ChainCallOptions},
    language_models::llm::LLM,
    prompt::HumanMessagePromptTemplate,
    template_jinja2,
    tools::SQLDatabase,
};

use super::{
    chain::SQLDatabaseChain,
    prompt::{DEFAULT_SQLSUFFIX, DEFAULT_SQLTEMPLATE},
    STOP_WORD,
};

pub struct SQLDatabaseChainBuilder<L>
where
    L: LLM + 'static,
{
    llm: Option<L>,
    options: Option<ChainCallOptions>,
    top_k: Option<usize>,
    database: Option<SQLDatabase>,
}

impl<L> SQLDatabaseChainBuilder<L>
where
    L: LLM + 'static,
{
    pub fn new() -> Self {
        Self {
            llm: None,
            options: None,
            top_k: None,
            database: None,
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

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn database(mut self, database: SQLDatabase) -> Self {
        self.database = Some(database);
        self
    }

    pub fn build(self) -> Result<SQLDatabaseChain, Box<dyn Error>> {
        let llm = self.llm.ok_or("LLM must be set")?;
        let top_k = self.top_k.ok_or("top_k must be set")?;
        let database = self.database.ok_or("Database must be set")?;

        let prompt = HumanMessagePromptTemplate::new(template_jinja2!(
            format!("{}{}", DEFAULT_SQLTEMPLATE, DEFAULT_SQLSUFFIX),
            "table_info",
            "top_k",
            "input"
        ));

        let llm_chain = match self.options {
            Some(options) => LLMChainBuilder::new()
                .prompt(prompt)
                .llm(llm)
                .options(options.with_stop_words(vec![STOP_WORD.to_string()]))
                .build()?,
            None => LLMChainBuilder::new()
                .prompt(prompt)
                .options(ChainCallOptions::default().with_stop_words(vec![STOP_WORD.to_string()]))
                .llm(llm)
                .build()?,
        };

        Ok(SQLDatabaseChain {
            llmchain: llm_chain,
            top_k,
            database,
        })
    }
}
