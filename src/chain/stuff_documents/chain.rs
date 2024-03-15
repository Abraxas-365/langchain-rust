use std::{error::Error, pin::Pin};

use async_trait::async_trait;
use futures::Stream;
use serde_json::Value;

use crate::{
    chain::{Chain, LLMChain},
    language_models::GenerateResult,
    prompt::PromptArgs,
    schemas::Document,
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
        let mut text = String::new();
        let doc_len = docs.len();
        for (k, doc) in docs.iter().enumerate() {
            text += &doc.page_content;
            if k != doc_len - 1 {
                text += &self.separator;
            }
        }
        text
    }
}

#[async_trait]
impl Chain for StuffDocument {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, Box<dyn Error>> {
        let docs = input_variables
            .get(&self.input_key)
            .ok_or("No documents found")?;

        let documents: Vec<Document> =
            serde_json::from_value(docs.clone()).map_err(|e| e.to_string())?;

        let mut input_values = input_variables.clone();
        input_values.insert(
            self.document_variable_name.clone(),
            Value::String(self.join_documents(documents)),
        );

        self.llm_chain.call(input_values).await
    }

    async fn stream(
        &self,
        _input_variables: PromptArgs,
    ) -> Result<
        Pin<Box<dyn Stream<Item = Result<serde_json::Value, Box<dyn Error + Send>>> + Send>>,
        Box<dyn Error>,
    > {
        self.llm_chain.stream(_input_variables).await
    }

    fn get_input_keys(&self) -> Vec<String> {
        vec![self.input_key.clone()]
    }
}
