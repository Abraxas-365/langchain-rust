use std::error::Error;

use async_trait::async_trait;
use serde_json::Value;

use crate::{
    chain::Chain,
    language_models::{GenerateResult, TokenUsage},
    prompt::PromptArgs,
};

pub struct SequentialChain {
    pub(crate) chains: Vec<Box<dyn Chain>>,
    pub(crate) outputs: Vec<String>,
}

impl SequentialChain {
    pub async fn execute(
        &self,
        input_variables: PromptArgs,
    ) -> Result<Vec<GenerateResult>, Box<dyn Error>> {
        let mut result: Vec<GenerateResult> = Vec::new();
        let mut variables = input_variables;
        for (i, chain) in (1..).zip(self.chains.iter()) {
            let output = chain.call(variables.clone()).await?;

            // Use i directly since it now starts from 1
            if i < self.chains.len() {
                variables.insert(
                    self.outputs[i].clone(),
                    Value::from(output.generation.clone()),
                );
            }
            log::debug!("Output: {:?}", output);
            result.push(output);
        }

        Ok(result)
    }
}

#[async_trait]
impl Chain for SequentialChain {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, Box<dyn Error>> {
        let mut token_usage: Option<TokenUsage> = None;
        let mut result = GenerateResult::default();
        let outputs = self.execute(input_variables).await?;
        for output in outputs.iter() {
            if let Some(token) = &output.tokens {
                match token_usage {
                    Some(usage) => token_usage = Some(usage.sum(&token)),
                    None => token_usage = Some(token.clone()),
                }
            }
            result.generation = output.generation.clone();
        }
        result.tokens = token_usage;
        Ok(result)
    }
    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>> {
        self.call(input_variables.clone())
            .await
            .map(|result| result.generation)
    }
    fn get_input_keys(&self) -> Vec<String> {
        self.outputs.clone()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        chain::{Chain, LLMChainBuilder},
        llm::openai::OpenAI,
        prompt_args, sequential_chain, template_fstring,
    };

    #[tokio::test]
    async fn test_sequential() {
        let llm = OpenAI::default();
        let chain1 = LLMChainBuilder::new()
            .prompt(template_fstring!(
                "dame un nombre para una tienda de {input}",
                "input"
            ))
            .llm(llm.clone())
            .build()
            .expect("Failed to build LLMChain");

        let chain2 = LLMChainBuilder::new()
            .prompt(template_fstring!(
                "dame un slogan para una tienda llamada {output}",
                "output"
            ))
            .llm(llm.clone())
            .build()
            .expect("Failed to build LLMChain");

        let chain = sequential_chain!(chain1, chain2);
        let result = chain.call(prompt_args! {"input"=>"medias"}).await;
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
