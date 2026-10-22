use std::{collections::HashMap, pin::Pin};

use async_trait::async_trait;
use futures::Stream;
use serde_json::{json, Value};

use crate::{
    language_models::{result_events, GenerateResult, LLMEvent},
    prompt::PromptArgs,
    schemas::StreamData,
};

use super::ChainError;

/// Stream returned by [`Chain::call_stream`].
pub type ChainEventStream<'a> =
    Pin<Box<dyn Stream<Item = Result<LLMEvent, ChainError>> + Send + 'a>>;

/// Boxes a chain event stream; also pins down the error type for `try_stream!` blocks.
pub(crate) fn box_chain_events<'a, S>(stream: S) -> ChainEventStream<'a>
where
    S: Stream<Item = Result<LLMEvent, ChainError>> + Send + 'a,
{
    Box::pin(stream)
}

pub(crate) const DEFAULT_OUTPUT_KEY: &str = "output";
pub(crate) const DEFAULT_RESULT_KEY: &str = "generate_result";

#[async_trait]
pub trait Chain: Sync + Send {
    /// Call the `Chain` and receive as output the result of the generation process along with
    /// additional information like token consumption. The input is a set of variables passed
    /// as a `PromptArgs` hashmap.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use crate::my_crate::{Chain, ConversationalChainBuilder, OpenAI, OpenAIModel, SimpleMemory, PromptArgs, prompt_args};
    /// # async {
    /// let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
    /// let memory = SimpleMemory::new();
    ///
    /// let chain = ConversationalChainBuilder::new()
    ///     .llm(llm)
    ///     .memory(memory.into())
    ///     .build().expect("Error building ConversationalChain");
    ///
    /// let input_variables = prompt_args! {
    ///     "input" => "Im from Peru",
    /// };
    ///
    /// match chain.call(input_variables).await {
    ///     Ok(result) => {
    ///         println!("Result: {:?}", result);
    ///     },
    ///     Err(e) => panic!("Error calling Chain: {:?}", e),
    /// };
    /// # };
    /// ```
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError>;

    /// Invoke the `Chain` and receive just the generation result as a String.
    /// The input is a set of variables passed as a `PromptArgs` hashmap.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use crate::my_crate::{Chain, ConversationalChainBuilder, OpenAI, OpenAIModel, SimpleMemory, PromptArgs, prompt_args};
    /// # async {
    /// let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
    /// let memory = SimpleMemory::new();
    ///
    /// let chain = ConversationalChainBuilder::new()
    ///     .llm(llm)
    ///     .memory(memory.into())
    ///     .build().expect("Error building ConversationalChain");
    ///
    /// let input_variables = prompt_args! {
    ///     "input" => "Im from Peru",
    /// };
    ///
    /// match chain.invoke(input_variables).await {
    ///     Ok(result) => {
    ///         println!("Result: {:?}", result);
    ///     },
    ///     Err(e) => panic!("Error invoking Chain: {:?}", e),
    /// };
    /// # };
    /// ```
    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, ChainError> {
        self.call(input_variables)
            .await
            .map(|result| result.generation)
    }

    /// Execute the `Chain` and return the result of the generation process
    /// along with additional information like token consumption formatted as a `HashMap`.
    /// The input is a set of variables passed as a `PromptArgs` hashmap.
    /// The key for the generated output is specified by the `get_output_keys`
    /// method (default key is `output`).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// # use crate::my_crate::{Chain, ConversationalChainBuilder, OpenAI, OpenAIModel, SimpleMemory, PromptArgs, prompt_args};
    /// # async {
    /// let llm = OpenAI::default().with_model(OpenAIModel::Gpt35);
    /// let memory = SimpleMemory::new();
    ///
    /// let chain = ConversationalChainBuilder::new()
    ///     .llm(llm)
    ///     .memory(memory.into())
    ///     .output_key("name")
    ///     .build().expect("Error building ConversationalChain");
    ///
    /// let input_variables = prompt_args! {
    ///     "input" => "Im from Peru",
    /// };
    ///
    /// match chain.execute(input_variables).await {
    ///     Ok(result) => {
    ///         println!("Result: {:?}", result);
    ///     },
    ///     Err(e) => panic!("Error executing Chain: {:?}", e),
    /// };
    /// # };
    /// ```
    async fn execute(
        &self,
        input_variables: PromptArgs,
    ) -> Result<HashMap<String, Value>, ChainError> {
        log::info!("Using default implementation");
        let result = self.call(input_variables.clone()).await?;
        let mut output = HashMap::new();
        let output_key = self
            .get_output_keys()
            .first()
            .unwrap_or(&DEFAULT_OUTPUT_KEY.to_string())
            .clone();
        output.insert(output_key, json!(result.generation));
        output.insert(DEFAULT_RESULT_KEY.to_string(), json!(result));
        Ok(output)
    }
    /// Stream the `Chain` and get an asynchronous stream of chain generations.
    /// The input is a set of variables passed as a `PromptArgs` hashmap.
    /// If the chain have memroy, the tream method will not be able to automaticaly
    /// set the memroy, bocause it will not know if the how to extract the output message
    /// out of the stram
    /// # Example
    ///
    /// ```rust,ignore
    /// # use futures::StreamExt;
    /// # use crate::my_crate::{Chain, LLMChainBuilder, OpenAI, fmt_message, fmt_template,
    /// #                      HumanMessagePromptTemplate, prompt_args, Message, template_fstring};
    /// # async {
    /// let open_ai = OpenAI::default();
    ///
    ///let prompt = message_formatter![
    ///fmt_message!(Message::new_system_message(
    ///"You are world class technical documentation writer."
    ///)),
    ///fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
    ///      "{input}", "input"
    ///)))
    ///];
    ///
    /// let chain = LLMChainBuilder::new()
    ///     .prompt(prompt)
    ///     .llm(open_ai.clone())
    ///     .build()
    ///     .unwrap();
    ///
    /// let mut stream = chain.stream(
    /// prompt_args! {
    /// "input" => "Who is the writer of 20,000 Leagues Under the Sea?"
    /// }).await.unwrap();
    ///
    /// while let Some(result) = stream.next().await {
    ///     match result {
    ///         Ok(value) => {
    ///                 println!("Content: {}", value.content);
    ///         },
    ///         Err(e) => panic!("Error invoking LLMChain: {:?}", e),
    ///     }
    /// };
    /// # };
    /// ```
    ///
    async fn stream(
        &self,
        _input_variables: PromptArgs,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, ChainError>> + Send>>, ChainError>
    {
        Err(ChainError::OtherError(
            "stream is not implemented for this chain".into(),
        ))
    }

    /// Like [`call`](Chain::call), but streams the LLM output as [`LLMEvent`]s: `Text`
    /// chunks, then one `Done` with the complete result.
    ///
    /// The default implementation runs `call` and emits the result in one chunk, so every
    /// chain supports it. [`LLMChain`](crate::chain::LLMChain) streams for real, which is
    /// what agents use to stream their answers.
    fn call_stream<'a>(&'a self, input_variables: PromptArgs) -> ChainEventStream<'a> {
        box_chain_events(async_stream::try_stream! {
            let result = self.call(input_variables).await?;
            for event in result_events(result) {
                yield event;
            }
        })
    }

    // Get the input keys of the prompt
    fn get_input_keys(&self) -> Vec<String> {
        log::info!("Using default implementation");
        vec![]
    }

    fn get_output_keys(&self) -> Vec<String> {
        log::info!("Using default implementation");
        vec![
            String::from(DEFAULT_OUTPUT_KEY),
            String::from(DEFAULT_RESULT_KEY),
        ]
    }
}

impl<C> From<C> for Box<dyn Chain>
where
    C: Chain + 'static,
{
    fn from(chain: C) -> Self {
        Box::new(chain)
    }
}
