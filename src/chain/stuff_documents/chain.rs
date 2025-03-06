use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde_json::Value;

use crate::{
    chain::{
        load_stuff_qa, options::ChainCallOptions, Chain, ChainError, LLMChain, StuffQAPromptBuilder,
    },
    language_models::{llm::LLM, GenerateResult},
    prompt::PromptArgs,
    schemas::{Document, StreamData},
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

    fn join_documents(&self, docs: Vec<Document>) -> String {
        docs.iter()
            .map(|doc| doc.page_content.clone())
            .collect::<Vec<_>>()
            .join(&self.separator)
    }

    ///Inly use thi if you use the deafult prompt
    pub fn qa_prompt_builder<'a>(&self) -> StuffQAPromptBuilder<'a> {
        StuffQAPromptBuilder::new()
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
        load_stuff_qa(llm, None)
    }

    /// load_stuff_qa_with_options return an instance of StuffDocument
    /// with a prompt desiged for question ansering
    ///
    /// # Example
    /// ```rust,ignore
    ///
    /// let llm = OpenAI::default();
    /// let chain = StuffDocument::load_stuff_qa_with_options(llm,ChainCallOptions::default());
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
    pub fn load_stuff_qa_with_options<L: LLM + 'static>(llm: L, opt: ChainCallOptions) -> Self {
        load_stuff_qa(llm, Some(opt))
    }
}

#[async_trait]
impl Chain for StuffDocument {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        let docs = input_variables
            .get(&self.input_key)
            .ok_or_else(|| ChainError::MissingInputVariable(self.input_key.clone()))?;

        let documents: Vec<Document> = serde_json::from_value(docs.clone()).map_err(|e| {
            ChainError::IncorrectInputVariable {
                source: e,
                expected_type: "Vec<Document>".to_string(),
            }
        })?;

        let mut input_values = input_variables.clone();
        input_values.insert(
            self.document_variable_name.clone(),
            Value::String(self.join_documents(documents)),
        );

        self.llm_chain.call(input_values).await
    }

    async fn stream(
        &self,
        input_variables: PromptArgs,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, ChainError>> + Send>>, ChainError>
    {
        let docs = input_variables
            .get(&self.input_key)
            .ok_or_else(|| ChainError::MissingInputVariable(self.input_key.clone()))?;

        let documents: Vec<Document> = serde_json::from_value(docs.clone()).map_err(|e| {
            ChainError::IncorrectInputVariable {
                source: e,
                expected_type: "Vec<Document>".to_string(),
            }
        })?;

        let mut input_values = input_variables.clone();
        input_values.insert(
            self.document_variable_name.clone(),
            Value::String(self.join_documents(documents)),
        );
        self.llm_chain.stream(input_values).await
    }

    fn get_input_keys(&self) -> Vec<String> {
        vec![self.input_key.clone()]
    }

    fn log_messages(&self, inputs: PromptArgs) -> Result<(), Box<dyn std::error::Error>> {
        self.llm_chain.log_messages(inputs)
    }
}
