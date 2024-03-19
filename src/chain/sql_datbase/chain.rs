use std::error::Error;

use async_trait::async_trait;
use serde_json::Value;

use crate::{
    chain::{chain_trait::Chain, llm_chain::LLMChain},
    language_models::{GenerateResult, TokenUsage},
    prompt::PromptArgs,
    prompt_args,
    tools::SQLDatabase,
};

use super::{
    QUERY_PREFIX_WITH, SQL_CHAIN_DEFAULT_INPUT_KEY_QUERY, SQL_CHAIN_DEFAULT_INPUT_KEY_TABLE_NAMES,
    STOP_WORD,
};

pub struct SqlChainPromptBuilder {
    input: String,
}
impl SqlChainPromptBuilder {
    pub fn new() -> Self {
        Self {
            input: "".to_string(),
        }
    }

    pub fn input<S: Into<String>>(mut self, input: S) -> Self {
        self.input = input.into();
        self
    }

    pub fn build(self) -> PromptArgs {
        prompt_args! {
          SQL_CHAIN_DEFAULT_INPUT_KEY_QUERY  => self.input,
        }
    }
}

pub struct SQLDatabaseChain {
    pub(crate) llmchain: LLMChain,
    pub(crate) top_k: usize,
    pub(crate) database: SQLDatabase,
}
impl SQLDatabaseChain {
    pub fn promp_builder() -> SqlChainPromptBuilder {
        SqlChainPromptBuilder::new()
    }
}

#[async_trait]
impl Chain for SQLDatabaseChain {
    fn get_input_keys(&self) -> Vec<String> {
        self.llmchain.get_input_keys()
    }

    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, Box<dyn Error>> {
        let mut token_usage: Option<TokenUsage> = None;

        let query = input_variables
            .get(SQL_CHAIN_DEFAULT_INPUT_KEY_QUERY)
            .ok_or("No query provided")?
            .to_string();

        let mut tables: Vec<String> = Vec::new();
        if let Some(value) = input_variables.get(SQL_CHAIN_DEFAULT_INPUT_KEY_TABLE_NAMES) {
            if let serde_json::Value::Array(array) = value {
                for item in array {
                    if let serde_json::Value::String(str) = item {
                        tables.push(str.clone());
                    }
                }
            }
        }

        let tables_info = self.database.table_info(&tables).await?;
        let mut llm_inputs = prompt_args! {
            "input"=> query.clone() + QUERY_PREFIX_WITH,
            "top_k"=> self.top_k,
            "dialect"=> self.database.dialect().to_string(),
            "table_info"=> tables_info,

        };

        let output = self.llmchain.call(llm_inputs.clone()).await?;
        if let Some(tokens) = output.tokens {
            token_usage = Some(tokens);
        }

        let sql_query = output.generation.trim();
        log::debug!("output: {:?}", sql_query);
        let query_result = self.database.query(sql_query).await?;

        llm_inputs.insert(
            "input".to_string(),
            Value::from(format!(
                "{}{}{}{}{}",
                &query, QUERY_PREFIX_WITH, sql_query, STOP_WORD, &query_result,
            )),
        );

        let output = self.llmchain.call(llm_inputs).await?;
        if let Some(tokens) = output.tokens {
            if let Some(general_result) = token_usage.as_mut() {
                general_result.completion_tokens += tokens.completion_tokens;
                general_result.total_tokens += tokens.total_tokens;
            }
        }

        let strs: Vec<&str> = output
            .generation
            .split("\n\n")
            .next()
            .unwrap_or("")
            .split("Answer:")
            .collect();
        let mut output = strs[0];
        if strs.len() > 1 {
            output = strs[1];
        }
        output = output.trim();
        Ok(GenerateResult {
            generation: output.to_string(),
            tokens: token_usage,
        })
    }
    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>> {
        let result = self.call(input_variables).await?;
        Ok(result.generation)
    }
}
