use std::{collections::HashSet, error::Error, pin::Pin};

use async_trait::async_trait;
use futures::Stream;
use futures_util::TryStreamExt;

use crate::{
    language_models::{llm::LLM, GenerateResult},
    output_parsers::{OutputParser, SimpleParser},
    schemas::{InputVariables, StreamData},
    template::PromptTemplate,
};

use super::{chain_trait::Chain, options::ChainCallOptions, ChainError};

pub struct LLMChainBuilder {
    prompt: Option<PromptTemplate>,
    llm: Option<Box<dyn LLM>>,
    output_key: Option<String>,
    options: Option<ChainCallOptions>,
    output_parser: Option<Box<dyn OutputParser>>,
}

impl LLMChainBuilder {
    pub fn new() -> Self {
        Self {
            prompt: None,
            llm: None,
            options: None,
            output_key: None,
            output_parser: None,
        }
    }
    pub fn options(mut self, options: ChainCallOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn prompt<P: Into<PromptTemplate>>(mut self, prompt: P) -> Self {
        self.prompt = Some(prompt.into());
        self
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

    pub fn build(self) -> Result<LLMChain, ChainError> {
        let prompt = self
            .prompt
            .ok_or_else(|| ChainError::MissingObject("Prompt must be set".into()))?;

        let mut llm = self
            .llm
            .ok_or_else(|| ChainError::MissingObject("LLM must be set".into()))?;

        if let Some(options) = self.options {
            let llm_options = ChainCallOptions::to_llm_options(options);
            llm.add_options(llm_options);
        }

        let chain = LLMChain {
            prompt,
            llm,
            output_parser: self
                .output_parser
                .unwrap_or_else(|| Box::new(SimpleParser::default())),
        };

        Ok(chain)
    }
}

impl Default for LLMChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct LLMChain {
    prompt: PromptTemplate,
    llm: Box<dyn LLM>,
    output_parser: Box<dyn OutputParser>,
}

#[async_trait]
impl Chain for LLMChain {
    fn get_input_keys(&self) -> HashSet<String> {
        self.prompt.variables()
    }

    async fn call(
        &self,
        input_variables: &mut InputVariables,
    ) -> Result<GenerateResult, ChainError> {
        let prompt = self.prompt.format(input_variables)?;
        let mut output = self.llm.generate(&prompt.to_messages()).await?;
        output.generation = self.output_parser.parse(&output.generation).await?;

        Ok(output)
    }

    async fn invoke(&self, input_variables: &mut InputVariables) -> Result<String, ChainError> {
        let prompt = self.prompt.format(input_variables)?;

        let output = self.llm.generate(&prompt.to_messages()).await?.generation;
        Ok(output)
    }

    async fn stream(
        &self,
        input_variables: &mut InputVariables,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, ChainError>> + Send>>, ChainError>
    {
        let prompt = self.prompt.format(input_variables)?;
        let llm_stream = self.llm.stream(&prompt.to_messages()).await?;

        // Map the errors from LLMError to ChainError
        let mapped_stream = llm_stream.map_err(ChainError::from);

        Ok(Box::pin(mapped_stream))
    }

    fn log_messages(&self, inputs: &InputVariables) -> Result<(), Box<dyn Error>> {
        let prompt = self.prompt.format(inputs)?;

        for message in prompt.to_messages() {
            log::debug!("{}:\n{}", message.message_type, message.content);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        chain::options::ChainCallOptions,
        llm::openai::{OpenAI, OpenAIModel},
        prompt_template,
        schemas::MessageType,
        template::MessageTemplate,
        text_replacements,
    };

    use super::*;

    #[tokio::test]
    #[ignore]
    async fn test_invoke_chain() {
        let mut input_variables: InputVariables = text_replacements! {
            "nombre" => "Juan",
        }
        .into();

        // Create an AI message prompt template
        let human_message_prompt =
            MessageTemplate::from_fstring(MessageType::HumanMessage, "Mi nombre es: {nombre} ");

        // Use the `message_formatter` macro to construct the formatter
        let prompt = prompt_template!(human_message_prompt);

        let options = ChainCallOptions::default();
        let llm = OpenAI::default().with_model(OpenAIModel::Gpt35.to_string());
        let chain = LLMChainBuilder::new()
            .prompt(prompt)
            .llm(llm)
            .options(options)
            .build()
            .expect("Failed to build LLMChain");

        // Execute `chain.invoke` and assert that it should succeed
        let result = chain.invoke(&mut input_variables).await;
        assert!(
            result.is_ok(),
            "Error invoking LLMChain: {:?}",
            result.err()
        )
    }
}
