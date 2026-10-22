use std::collections::HashMap;

use serde_json::{json, Value};

use crate::{
    chain::{Chain, ChainError, ChainEventStream, StuffDocument, DEFAULT_RESULT_KEY},
    language_models::{llm::LLM, GenerateResult},
    prompt::PromptArgs,
    schemas::{messages::value_to_text, Document, Retriever},
};

const RETRIEVAL_QA_DEFAULT_INPUT_KEY: &str = "query";
const RETRIEVAL_QA_DEFAULT_OUTPUT_KEY: &str = "output";
/// Output key of [`RetrievalQA::execute`] holding the retrieved documents, when
/// [`with_return_source_documents`](RetrievalQA::with_return_source_documents) is on.
pub const RETRIEVAL_QA_SOURCE_DOCUMENTS_KEY: &str = "source_documents";

/// Question answering over a [`Retriever`]: fetches the documents relevant to the query,
/// then hands them with the question to a document-combining chain.
///
/// Stateless, unlike [`ConversationalRetrieverChain`](crate::chain::ConversationalRetrieverChain):
/// no memory, no question rephrasing. The combining chain receives `input_documents` and
/// `question`, so [`StuffDocument`], [`RefineDocuments`](crate::chain::RefineDocuments) and
/// [`MapReduceDocuments`](crate::chain::MapReduceDocuments) QA chains all fit.
///
/// ```rust,ignore
/// let qa = RetrievalQA::from_llm(OpenAI::default(), store_retriever);
/// let answer = qa.invoke(prompt_args! { "query" => "What is our refund policy?" }).await?;
/// ```
pub struct RetrievalQA {
    retriever: Box<dyn Retriever>,
    combine_documents_chain: Box<dyn Chain>,
    input_key: String,
    output_key: String,
    return_source_documents: bool,
}

impl RetrievalQA {
    pub fn new<R: Into<Box<dyn Retriever>>, C: Into<Box<dyn Chain>>>(
        retriever: R,
        combine_documents_chain: C,
    ) -> Self {
        Self {
            retriever: retriever.into(),
            combine_documents_chain: combine_documents_chain.into(),
            input_key: RETRIEVAL_QA_DEFAULT_INPUT_KEY.into(),
            output_key: RETRIEVAL_QA_DEFAULT_OUTPUT_KEY.into(),
            return_source_documents: false,
        }
    }

    /// Stuffs every retrieved document into one QA prompt.
    pub fn from_llm<L: Into<Box<dyn LLM>>, R: Into<Box<dyn Retriever>>>(
        llm: L,
        retriever: R,
    ) -> Self {
        Self::new(retriever, StuffDocument::load_stuff_qa(llm))
    }

    /// Input key holding the query (default `query`).
    pub fn with_input_key<S: Into<String>>(mut self, key: S) -> Self {
        self.input_key = key.into();
        self
    }

    /// Output key of the answer in [`execute`](Chain::execute) (default `output`).
    pub fn with_output_key<S: Into<String>>(mut self, key: S) -> Self {
        self.output_key = key.into();
        self
    }

    /// Adds the retrieved documents to [`execute`](Chain::execute)'s output under
    /// [`RETRIEVAL_QA_SOURCE_DOCUMENTS_KEY`].
    pub fn with_return_source_documents(mut self, yes: bool) -> Self {
        self.return_source_documents = yes;
        self
    }

    async fn retrieve(
        &self,
        input_variables: &PromptArgs,
    ) -> Result<(PromptArgs, Vec<Document>), ChainError> {
        let query = input_variables
            .get(&self.input_key)
            .map(value_to_text)
            .ok_or_else(|| ChainError::MissingInputVariable(self.input_key.clone()))?;
        let documents = self
            .retriever
            .get_relevant_documents(&query)
            .await
            .map_err(|e| ChainError::RetrieverError(e.to_string()))?;
        let inputs = crate::prompt_args! {
            "question" => query,
            "input_documents" => documents,
        };
        Ok((inputs, documents))
    }
}

#[async_trait::async_trait]
impl Chain for RetrievalQA {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        let (inputs, _) = self.retrieve(&input_variables).await?;
        self.combine_documents_chain.call(inputs).await
    }

    async fn execute(
        &self,
        input_variables: PromptArgs,
    ) -> Result<HashMap<String, Value>, ChainError> {
        let (inputs, documents) = self.retrieve(&input_variables).await?;
        let result = self.combine_documents_chain.call(inputs).await?;
        let mut output = HashMap::new();
        output.insert(self.output_key.clone(), json!(result.generation));
        output.insert(DEFAULT_RESULT_KEY.to_string(), json!(result));
        if self.return_source_documents {
            output.insert(
                RETRIEVAL_QA_SOURCE_DOCUMENTS_KEY.to_string(),
                json!(documents),
            );
        }
        Ok(output)
    }

    /// Retrieves, then streams the combining chain's answer.
    fn call_stream<'a>(&'a self, input_variables: PromptArgs) -> ChainEventStream<'a> {
        use futures::StreamExt;
        Box::pin(
            futures::stream::once(async move {
                match self.retrieve(&input_variables).await {
                    Ok((inputs, _)) => self.combine_documents_chain.call_stream(inputs),
                    Err(e) => Box::pin(futures::stream::iter([Err(e)])) as ChainEventStream<'a>,
                }
            })
            .flatten(),
        )
    }

    fn get_input_keys(&self) -> Vec<String> {
        vec![self.input_key.clone()]
    }

    fn get_output_keys(&self) -> Vec<String> {
        let mut keys = vec![self.output_key.clone(), DEFAULT_RESULT_KEY.to_string()];
        if self.return_source_documents {
            keys.push(RETRIEVAL_QA_SOURCE_DOCUMENTS_KEY.to_string());
        }
        keys
    }
}
