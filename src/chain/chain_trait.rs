use std::{collections::HashMap, error::Error, pin::Pin};

use async_trait::async_trait;
use futures::Stream;
use serde_json::{json, Value};

use crate::{language_models::GenerateResult, prompt::PromptArgs};

pub(crate) const DEFAULT_OUTPUT_KEY: &str = "output";
pub(crate) const DEFAULT_RESULT_KEY: &str = "generate_result";

#[async_trait]
pub trait Chain: Sync + Send {
    //Call will return te output of the LLM plus the other metadata
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, Box<dyn Error>>;

    //invoke its an eazy way to just return the string
    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>>;

    //Execute will return the ouptut of the llm as a hasmap with a default output key.
    //Usefull when you need to contatenates chain outputs.
    //By default the key is output
    async fn execute(
        &self,
        input_variables: PromptArgs,
    ) -> Result<HashMap<String, Value>, Box<dyn Error>> {
        log::info!("Using defualt implementation");
        let result = self.call(input_variables.clone()).await?;
        let mut output = HashMap::new();
        let output_key = self
            .get_output_keys()
            .get(0)
            .unwrap_or(&DEFAULT_OUTPUT_KEY.to_string())
            .clone();
        output.insert(output_key, json!(result.generation));
        output.insert(DEFAULT_RESULT_KEY.to_string(), json!(result));
        Ok(output)
    }

    async fn stream(
        &self,
        _input_variables: PromptArgs,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<serde_json::Value, Box<dyn Error + Send>>> + Send>>,
        Box<dyn Error>,
    > {
        log::warn!("stream not implemented for this chain");
        unimplemented!()
    }

    // Get the input keys of the prompt
    fn get_input_keys(&self) -> Vec<String> {
        log::info!("Using defualt implementation");
        return vec![];
    }

    fn get_output_keys(&self) -> Vec<String> {
        log::info!("Using defualt implementation");
        return vec![String::from(DEFAULT_OUTPUT_KEY)];
    }
}
