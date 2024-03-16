use std::{
    collections::{HashMap, HashSet},
    error::Error,
};

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::{
    chain::{Chain, DEFAULT_OUTPUT_KEY, DEFAULT_RESULT_KEY},
    language_models::{GenerateResult, TokenUsage},
    prompt::PromptArgs,
};

//THIS IS EXPERIMENTAL
pub struct SequentialChain {
    pub(crate) chains: Vec<Box<dyn Chain>>,
    pub(crate) input_keys: HashSet<String>,
    pub(crate) outputs: HashSet<String>,
}

#[async_trait]
impl Chain for SequentialChain {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, Box<dyn Error>> {
        let output = self.execute(input_variables).await?;
        let result = output
            .get(DEFAULT_RESULT_KEY)
            .ok_or("No result key")?
            .clone();
        let result: GenerateResult = serde_json::from_value(result)?;
        Ok(result)
    }
    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>> {
        self.call(input_variables.clone())
            .await
            .map(|result| result.generation)
    }
    fn get_input_keys(&self) -> Vec<String> {
        self.outputs.iter().cloned().collect()
    }

    async fn execute(
        &self,
        input_variables: PromptArgs,
    ) -> Result<HashMap<String, Value>, Box<dyn Error>> {
        let mut input_variables = input_variables;
        let mut final_token_usage: Option<TokenUsage> = None;
        let mut output_result = HashMap::new();
        let mut final_result = GenerateResult::default();
        for chain in self.chains.iter() {
            let output = chain.execute(input_variables.clone()).await?;
            //Get the oput key for the chain result
            let output_key = chain
                .get_output_keys()
                .get(0)
                .unwrap_or(&DEFAULT_OUTPUT_KEY.to_string())
                .clone();
            //Get the ouput complete result
            let result = output
                .get(DEFAULT_RESULT_KEY)
                .unwrap_or(&json!(GenerateResult::default()))
                .clone();
            let result: GenerateResult = serde_json::from_value(result)?;
            log::debug!("{}", result.generation);
            //Insert the output chain to the final output
            output_result.insert(output_key.clone(), json!(result.generation.clone()));
            input_variables.insert(output_key, json!(result.generation.clone()));

            //add the generation to keep track of the final generation
            final_result.generation = result.generation;
            //Add to the token if it exist
            if let Some(token) = &result.tokens {
                match final_token_usage {
                    Some(token_usage) => {
                        final_token_usage = Some(token_usage.sum(&token));
                    }
                    None => {
                        final_token_usage = Some(token.clone());
                    }
                }
            }
        }

        //add the filan token count to the result
        final_result.tokens = final_token_usage;
        output_result.insert(DEFAULT_RESULT_KEY.to_string(), json!(final_result));
        Ok(output_result)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        chain::{Chain, LLMChainBuilder},
        llm::openai::{OpenAI, OpenAIConfig},
        prompt_args, sequential_chain, template_fstring,
    };

    #[tokio::test]
    async fn test_sequential() {
        let llm = OpenAI::new(OpenAIConfig::default());
        let chain1 = LLMChainBuilder::new()
            .prompt(template_fstring!(
                "dame un nombre para una tienda de {input}",
                "input"
            ))
            .llm(llm.clone())
            .output_key("nombre")
            .build()
            .expect("Failed to build LLMChain");

        let chain2 = LLMChainBuilder::new()
            .prompt(template_fstring!(
                "dame un slogan para una tienda llamada {nombre},tiene que incluir la palabra {palabra}",
                "nombre",
            "palabra"
            ))
            .llm(llm.clone())
            .output_key("slogan")
            .build()
            .expect("Failed to build LLMChain");

        let chain = sequential_chain!(chain1, chain2);
        let result = chain
            .execute(prompt_args! {"input"=>"medias","palabra"=>"arroz"})
            .await;
        assert!(
            result.is_ok(),
            "Expected `chain.call` to succeed, but it failed with error: {:?}",
            result.err()
        );

        if let Ok(output) = result {
            println!("{:?}", output);
        }
    }
}
