use std::{collections::HashMap, pin::Pin};

use async_trait::async_trait;
use derive_new::new;
use futures::Stream;

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

#[derive(Clone, new)]
pub struct StuffQA {
    documents: Vec<Document>,
    input: HashMap<String, String>,
}

impl PromptArgs for StuffQA {
    fn contains_key(&self, key: &str) -> bool {
        self.input.contains_key(key)
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.input.get(key).map(|s| s.as_str())
    }
    fn insert(&mut self, key: String, value: String) -> Option<String> {
        self.input.insert(key.to_string(), value.to_string())
    }

    fn iter(&self) -> Box<dyn Iterator<Item = (&String, &String)> + '_> {
        Box::new(self.input.iter())
    }
}

pub struct StuffDocument {
    llm_chain: LLMChain<StuffQA>,
    input_key: String,
    document_variable_name: String,
    separator: String,
}

impl StuffDocument {
    pub fn new(llm_chain: LLMChain<StuffQA>) -> Self {
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
    pub fn qa_prompt_builder(&self) -> StuffQAPromptBuilder {
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
impl Chain<StuffQA> for StuffDocument {
    async fn call(&self, input_variables: &mut StuffQA) -> Result<GenerateResult, ChainError> {
        let docs = input_variables
            .get(&self.input_key)
            .ok_or_else(|| ChainError::MissingInputVariable(self.input_key.clone()))?;

        let documents: Vec<Document> =
            serde_json::from_str(docs).map_err(|e| ChainError::IncorrectInputVariable {
                source: e,
                expected_type: "Vec<Document>".to_string(),
            })?;

        input_variables.insert(
            self.document_variable_name.clone(),
            self.join_documents(documents),
        );

        self.llm_chain.call(input_variables).await
    }

    async fn stream(
        &self,
        input_variables: &mut StuffQA,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, ChainError>> + Send>>, ChainError>
    {
        let docs = input_variables
            .get(&self.input_key)
            .ok_or_else(|| ChainError::MissingInputVariable(self.input_key.clone()))?;

        let documents: Vec<Document> =
            serde_json::from_str(docs).map_err(|e| ChainError::IncorrectInputVariable {
                source: e,
                expected_type: "Vec<Document>".to_string(),
            })?;

        input_variables.insert(
            self.document_variable_name.clone(),
            self.join_documents(documents),
        );

        self.llm_chain.stream(input_variables).await
    }

    fn get_input_keys(&self) -> Vec<String> {
        vec![self.input_key.clone()]
    }

    fn log_messages(&self, inputs: &StuffQA) -> Result<(), Box<dyn std::error::Error>> {
        self.llm_chain.log_messages(inputs)
    }
}
