use std::{collections::HashSet, pin::Pin};

use async_trait::async_trait;
use futures::Stream;

use crate::{
    chain::{load_stuff_qa, Chain, ChainError, LLMChain},
    language_models::{llm::LLM, GenerateResult},
    schemas::{InputVariables, StreamData},
};

const COMBINE_DOCUMENTS_DEFAULT_INPUT_KEY: &str = "input_documents";
const COMBINE_DOCUMENTS_DEFAULT_OUTPUT_KEY: &str = "text";
const COMBINE_DOCUMENTS_DEFAULT_DOCUMENT_VARIABLE_NAME: &str = "context";
const STUFF_DOCUMENTS_DEFAULT_SEPARATOR: &str = "\n\n";

pub struct StuffDocument {
    llm_chain: LLMChain,
    input_key: String,
    document_variable_name: String,
    separator: String,
}

impl StuffDocument {
    pub fn new(llm_chain: LLMChain) -> Self {
        Self {
            llm_chain,
            input_key: COMBINE_DOCUMENTS_DEFAULT_INPUT_KEY.to_string(),
            document_variable_name: COMBINE_DOCUMENTS_DEFAULT_DOCUMENT_VARIABLE_NAME.to_string(),
            separator: STUFF_DOCUMENTS_DEFAULT_SEPARATOR.to_string(),
        }
    }

    /// load_stuff_qa return an instance of StuffDocument
    /// with a prompt desiged for question ansering
    ///
    /// # Example
    /// ```rust,ignore
    ///
    /// let llm = OpenAI::default();
    /// let chain = StuffDocument::load_stuff_qa(llm);
    ///
    /// let input = chain
    /// .qa_prompt_builder()
    /// .documents(&[
    /// Document::new(format!(
    /// "\nQuestion: {}\nAnswer: {}\n",
    /// "Which is the favorite text editor of luis", "Nvim"
    /// )),
    /// Document::new(format!(
    /// "\nQuestion: {}\nAnswer: {}\n",
    /// "How old is Luis", "24"
    /// )),
    /// ])
    /// .question("How old is luis and whats his favorite text editor")
    /// .build();
    ///
    /// let ouput = chain.invoke(input).await.unwrap();
    ///
    /// println!("{}", ouput);
    /// ```
    ///
    pub fn load_stuff_qa<L: Into<Box<dyn LLM>>>(llm: L) -> Self {
        load_stuff_qa(llm)
    }
}

#[async_trait]
impl Chain for StuffDocument {
    async fn call(
        &self,
        input_variables: &mut InputVariables,
    ) -> Result<GenerateResult, ChainError> {
        self.llm_chain.call(input_variables).await
    }

    async fn stream(
        &self,
        input_variables: &mut InputVariables,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, ChainError>> + Send>>, ChainError>
    {
        self.llm_chain.stream(input_variables).await
    }

    fn get_input_keys(&self) -> HashSet<String> {
        [self.input_key.clone()].into_iter().collect()
    }

    fn log_messages(&self, inputs: &InputVariables) -> Result<(), Box<dyn std::error::Error>> {
        self.llm_chain.log_messages(inputs)
    }
}
