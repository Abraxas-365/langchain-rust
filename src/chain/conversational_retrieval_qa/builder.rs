use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{
    chain::{
        Chain, ChainError, CondenseQuestionGeneratorChain, StuffDocumentBuilder, StuffQA,
        DEFAULT_OUTPUT_KEY,
    },
    language_models::llm::LLM,
    memory::SimpleMemory,
    prompt::FormatPrompter,
    schemas::{BaseMemory, Retriever},
};

use super::ConversationalRetrieverChain;

const CONVERSATIONAL_RETRIEVAL_QA_DEFAULT_INPUT_KEY: &str = "question";

///Conversation Retriever Chain Builder
/// # Usage
/// ## Convensional way
/// ```rust,ignore
/// let chain = ConversationalRetrieverChainBuilder::new()
///     .llm(llm)
///     .rephrase_question(true)
///     .retriever(RetrieverMock {})
///     .memory(SimpleMemory::new().into())
///     .build()
///     .expect("Error building ConversationalChain");
///
/// ```
/// ## Custom way
/// ```rust,ignore
///
/// let llm = Box::new(OpenAI::default().with_model(OpenAIModel::Gpt35.to_string()));
/// let combine_documents_chain = StuffDocument::load_stuff_qa(llm.clone_box());
//  let condense_question_chain = CondenseQuestionGeneratorChain::new(llm.clone_box());
/// let chain = ConversationalRetrieverChainBuilder::new()
///     .rephrase_question(true)
///     .combine_documents_chain(Box::new(combine_documents_chain))
///     .condense_question_chain(Box::new(condense_question_chain))
///     .retriever(RetrieverMock {})
///     .memory(SimpleMemory::new().into())
///     .build()
///     .expect("Error building ConversationalChain");
/// ```
///
pub struct ConversationalRetrieverChainBuilder {
    llm: Option<Box<dyn LLM>>,
    retriever: Option<Box<dyn Retriever>>,
    memory: Option<Arc<Mutex<dyn BaseMemory>>>,
    combine_documents_chain: Option<Box<dyn Chain<StuffQA>>>,
    condense_question_chain: Option<Box<dyn Chain<StuffQA>>>,
    prompt: Option<Box<dyn FormatPrompter<StuffQA>>>,
    rephrase_question: bool,
    return_source_documents: bool,
    input_key: String,
    output_key: String,
}
impl ConversationalRetrieverChainBuilder {
    pub fn new() -> Self {
        ConversationalRetrieverChainBuilder {
            llm: None,
            retriever: None,
            memory: None,
            combine_documents_chain: None,
            condense_question_chain: None,
            prompt: None,
            rephrase_question: true,
            return_source_documents: true,
            input_key: CONVERSATIONAL_RETRIEVAL_QA_DEFAULT_INPUT_KEY.to_string(),
            output_key: DEFAULT_OUTPUT_KEY.to_string(),
        }
    }

    pub fn retriever<R: Into<Box<dyn Retriever>>>(mut self, retriever: R) -> Self {
        self.retriever = Some(retriever.into());
        self
    }

    ///If you want to add a custom prompt,keep in mind which variables are obligatory.
    pub fn prompt<P: Into<Box<dyn FormatPrompter<StuffQA>>>>(mut self, prompt: P) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    pub fn input_key<S: Into<String>>(mut self, input_key: S) -> Self {
        self.input_key = input_key.into();
        self
    }

    pub fn memory(mut self, memory: Arc<Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn llm<L: Into<Box<dyn LLM>>>(mut self, llm: L) -> Self {
        self.llm = Some(llm.into());
        self
    }

    ///Chain designed to take the documents and the question and generate an output
    pub fn combine_documents_chain<C: Into<Box<dyn Chain<StuffQA>>>>(
        mut self,
        combine_documents_chain: C,
    ) -> Self {
        self.combine_documents_chain = Some(combine_documents_chain.into());
        self
    }

    ///Chain designed to reformulate the question based on the cat history
    pub fn condense_question_chain<C: Into<Box<dyn Chain<StuffQA>>>>(
        mut self,
        condense_question_chain: C,
    ) -> Self {
        self.condense_question_chain = Some(condense_question_chain.into());
        self
    }

    pub fn rephrase_question(mut self, rephrase_question: bool) -> Self {
        self.rephrase_question = rephrase_question;
        self
    }

    pub fn return_source_documents(mut self, return_source_documents: bool) -> Self {
        self.return_source_documents = return_source_documents;
        self
    }

    pub fn build(mut self) -> Result<ConversationalRetrieverChain, ChainError> {
        if let Some(llm) = self.llm {
            let combine_documents_chain = {
                let mut builder = StuffDocumentBuilder::new().llm(llm.clone_box());
                if let Some(prompt) = self.prompt {
                    builder = builder.prompt(prompt);
                }
                builder.build()?
            };
            let condense_question_chain = CondenseQuestionGeneratorChain::new(llm.clone_box());
            self.combine_documents_chain = Some(Box::new(combine_documents_chain));
            self.condense_question_chain = Some(Box::new(condense_question_chain));
        }

        let retriever = self
            .retriever
            .ok_or_else(|| ChainError::MissingObject("Retriever must be set".into()))?;

        let memory = self
            .memory
            .unwrap_or_else(|| Arc::new(Mutex::new(SimpleMemory::new())));

        let combine_documents_chain = self.combine_documents_chain.ok_or_else(|| {
            ChainError::MissingObject(
                "Combine documents chain must be set or llm must be set".into(),
            )
        })?;
        let condense_question_chain = self.condense_question_chain.ok_or_else(|| {
            ChainError::MissingObject(
                "Condense question chain must be set or llm must be set".into(),
            )
        })?;
        Ok(ConversationalRetrieverChain {
            retriever,
            memory,
            combine_documents_chain,
            condense_question_chain,
            rephrase_question: self.rephrase_question,
            return_source_documents: self.return_source_documents,
            input_key: self.input_key,
            output_key: self.output_key,
        })
    }
}

impl Default for ConversationalRetrieverChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}
