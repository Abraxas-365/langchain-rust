use std::{collections::HashMap, error::Error};

use async_trait::async_trait;

use crate::{language_models::GenerateResult, prompt::PromptArgs};

pub(crate) const DEFAULT_OUTPUT_KEY: &str = "output";

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
    ) -> Result<HashMap<String, String>, Box<dyn Error>> {
        log::info!("Using defualt implementation");
        let result = self.invoke(input_variables.clone()).await?;
        let output = self
            .get_output_keys()
            .first()
            .unwrap_or(&String::from(DEFAULT_OUTPUT_KEY))
            .clone();

        Ok(HashMap::from([(output, result)]))
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
