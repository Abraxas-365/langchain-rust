use std::{error::Error, pin::Pin};

use async_trait::async_trait;
use futures::Stream;

use crate::{
    language_models::{llm::LLM, GenerateResult},
    prompt::{FormatPrompter, PromptArgs},
};

use super::{chain_trait::Chain, options::ChainCallOptions};

pub struct LLMChainBuilder {
    prompt: Option<Box<dyn FormatPrompter>>,
    llm: Option<Box<dyn LLM>>,
    output_key: Option<String>,
    options: Option<ChainCallOptions>,
}

impl LLMChainBuilder {
    pub fn new() -> Self {
        Self {
            prompt: None,
            llm: None,
            options: None,
            output_key: None,
        }
    }
    pub fn options(mut self, options: ChainCallOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn prompt<P>(mut self, prompt: P) -> Self
    where
        P: FormatPrompter + 'static,
    {
        self.prompt = Some(Box::new(prompt));
        self
    }

    pub fn llm<L>(mut self, llm: L) -> Self
    where
        L: LLM + 'static,
    {
        self.llm = Some(Box::new(llm));
        self
    }

    pub fn output_key<S: Into<String>>(mut self, output_key: S) -> Self {
        self.output_key = Some(output_key.into());
        self
    }

    pub fn build(self) -> Result<LLMChain, Box<dyn Error>> {
        let prompt = self.prompt.ok_or("Prompt must be set")?;
        let mut llm = self.llm.ok_or("LLM must be set")?;
        if let Some(options) = self.options {
            let llm_options = ChainCallOptions::to_llm_options(options);
            llm.with_options(llm_options);
        }

        let chain = LLMChain {
            prompt,
            llm,
            output_key: self.output_key.unwrap_or("output".to_string()),
        };

        Ok(chain)
    }
}

pub struct LLMChain {
    prompt: Box<dyn FormatPrompter>,
    llm: Box<dyn LLM>,
    output_key: String,
}

#[async_trait]
impl Chain for LLMChain {
    fn get_input_keys(&self) -> Vec<String> {
        return self.prompt.get_input_variables();
    }

    fn get_output_keys(&self) -> Vec<String> {
        vec![self.output_key.clone()]
    }

    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, Box<dyn Error>> {
        let prompt = self.prompt.format_prompt(input_variables.clone())?;
        log::debug!("Prompt: {:?}", prompt);
        let output = self.llm.generate(&prompt.to_chat_messages()).await?;
        Ok(output)
    }

    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>> {
        let prompt = self.prompt.format_prompt(input_variables.clone())?;
        log::debug!("Prompt: {:?}", prompt);
        let output = self
            .llm
            .generate(&prompt.to_chat_messages())
            .await?
            .generation;
        Ok(output)
    }

    async fn stream(
        &self,
        input_variables: PromptArgs,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<serde_json::Value, Box<dyn Error + Send>>> + Send>>,
        Box<dyn Error>,
    > {
        let prompt = self.prompt.format_prompt(input_variables.clone())?;
        log::debug!("Prompt: {:?}", prompt);
        self.llm.stream(&prompt.to_chat_messages()).await
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        chain::options::ChainCallOptions,
        llm::openai::{OpenAI, OpenAIModel},
        message_formatter,
        prompt::{HumanMessagePromptTemplate, MessageOrTemplate},
        prompt_args, template_fstring,
    };

    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_invoke_chain() {
        // Create an AI message prompt template
        let human_message_prompt = HumanMessagePromptTemplate::new(template_fstring!(
            "Mi nombre es: {nombre} ",
            "nombre",
        ));

        // Use the `message_formatter` macro to construct the formatter
        let formatter =
            message_formatter![MessageOrTemplate::Template(human_message_prompt.into()),];

        let options = ChainCallOptions::default();
        let llm = OpenAI::default().with_model(OpenAIModel::Gpt35.to_string());
        let chain = LLMChainBuilder::new()
            .prompt(formatter)
            .llm(llm)
            .options(options)
            .build()
            .expect("Failed to build LLMChain");

        let input_variables = prompt_args! {
            "nombre" => "luis",

        };
        // Execute `chain.invoke` and assert that it should succeed
        let result = chain.invoke(input_variables).await;
        assert!(
            result.is_ok(),
            "Error invoking LLMChain: {:?}",
            result.err()
        )
    }
}
