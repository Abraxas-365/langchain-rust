use std::{collections::HashSet, pin::Pin};

use async_trait::async_trait;
use futures::Stream;

use crate::{
    chain::{chain_trait::Chain, llm_chain::LLMChain, ChainError},
    input_variables,
    language_models::{GenerateResult, TokenUsage},
    schemas::{InputVariables, StreamData},
    tools::SQLDatabase,
};

use super::{
    QUERY_PREFIX_WITH, SQL_CHAIN_DEFAULT_INPUT_KEY_QUERY, SQL_CHAIN_DEFAULT_INPUT_KEY_TABLE_NAMES,
    STOP_WORD,
};

pub struct SqlChainPromptBuilder {
    query: String,
}
impl SqlChainPromptBuilder {
    pub fn new() -> Self {
        Self {
            query: "".to_string(),
        }
    }

    pub fn query<S: Into<String>>(mut self, input: S) -> Self {
        self.query = input.into();
        self
    }

    pub fn build(self) -> InputVariables {
        input_variables! {
          SQL_CHAIN_DEFAULT_INPUT_KEY_QUERY  => self.query,
        }
    }
}

impl Default for SqlChainPromptBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct SQLDatabaseChain {
    pub(crate) llmchain: LLMChain,
    pub(crate) top_k: usize,
    pub(crate) database: SQLDatabase,
}

/// SQLChain let you interact with a db in human lenguage
///
/// The input variable name is `query`.
/// Example
/// ```rust,ignore
/// # async {
/// let options = ChainCallOptions::default();
/// let llm = OpenAI::default();
///
/// let db = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
/// let engine = PostgreSQLEngine::new(&db).await.unwrap();
/// let db = SQLDatabaseBuilder::new(engine).build().await.unwrap();
/// let chain = SQLDatabaseChainBuilder::new()
///     .llm(llm)
///     .top_k(4)
///     .database(db)
///     .options(options)
///     .build()
///     .expect("Failed to build LLMChain");
///
/// let input_variables = prompt_args! {
///     "query" => "Whats the phone number of luis"
///   };
///   //OR
/// let input_variables = chain.prompt_builder()
///     .query("Whats the phone number of luis")
///     .build();
/// match chain.invoke(input_variables).await {
///    Ok(result) => {
///     println!("Result: {:?}", result);
/// }
/// Err(e) => panic!("Error invoking LLMChain: {:?}", e),
/// }
///
/// }
/// ```
impl SQLDatabaseChain {
    pub fn prompt_builder(&self) -> SqlChainPromptBuilder {
        SqlChainPromptBuilder::new()
    }

    async fn call_builder_chains(
        &self,
        input_variables: &InputVariables,
    ) -> Result<(InputVariables, Option<TokenUsage>), ChainError> {
        let mut token_usage: Option<TokenUsage> = None;

        let query = input_variables
            .get(SQL_CHAIN_DEFAULT_INPUT_KEY_QUERY)
            .ok_or_else(|| {
                ChainError::MissingInputVariable(SQL_CHAIN_DEFAULT_INPUT_KEY_QUERY.to_string())
            })?
            .to_string();

        let mut tables: Vec<String> = Vec::new();
        if let Some(array) = input_variables.get(SQL_CHAIN_DEFAULT_INPUT_KEY_TABLE_NAMES) {
            for item in array.split(",") {
                tables.push(item.to_string());
            }
        } // WONTFIX: Possibly broken

        let tables_info = self
            .database
            .table_info(&tables)
            .await
            .map_err(|e| ChainError::DatabaseError(e.to_string()))?;

        let mut llm_inputs = input_variables! {
            "input"=> query.clone() + QUERY_PREFIX_WITH,
            "top_k"=> self.top_k,
            "dialect"=> self.database.dialect().to_string(),
            "table_info"=> tables_info,

        };

        let output = self.llmchain.call(&mut llm_inputs).await?;
        if let Some(tokens) = output.tokens {
            token_usage = Some(tokens);
        }

        let sql_query = output.generation.trim();
        let query_result = self
            .database
            .query(sql_query)
            .await
            .map_err(|e| ChainError::DatabaseError(e.to_string()))?;

        llm_inputs.insert(
            "input".to_string(),
            format!(
                "{}{}{}{}{}",
                &query, QUERY_PREFIX_WITH, sql_query, STOP_WORD, &query_result,
            ),
        );
        Ok((llm_inputs, token_usage))
    }
}

#[async_trait]
impl Chain for SQLDatabaseChain {
    fn get_input_keys(&self) -> HashSet<String> {
        self.llmchain.get_input_keys()
    }

    async fn call(
        &self,
        input_variables: &mut InputVariables,
    ) -> Result<GenerateResult, ChainError> {
        let (mut llm_inputs, mut token_usage) = self.call_builder_chains(input_variables).await?;
        let output = self.llmchain.call(&mut llm_inputs).await?;
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

    async fn invoke(&self, input_variables: &mut InputVariables) -> Result<String, ChainError> {
        let result = self.call(input_variables).await?;
        Ok(result.generation)
    }

    async fn stream(
        &self,
        input_variables: &mut InputVariables,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, ChainError>> + Send>>, ChainError>
    {
        let (mut llm_inputs, _) = self.call_builder_chains(input_variables).await?;

        self.llmchain.stream(&mut llm_inputs).await
    }

    fn log_messages(&self, inputs: &InputVariables) -> Result<(), Box<dyn std::error::Error>> {
        self.llmchain.log_messages(inputs)
    }
}
