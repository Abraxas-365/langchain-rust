use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{
    chain::{Chain, ChainError, CondenseQuetionGeneratorChain, StuffDocument, DEFAULT_OUTPUT_KEY},
    language_models::llm::LLM,
    memory::SimpleMemory,
    schemas::{BaseMemory, Retriever},
};

use super::ConversationalRetriverChain;

const CONVERSATIONAL_RETRIEVAL_QA_DEFAULT_INPUT_KEY: &str = "question";

///Conversation Retriver Chain Builder
/// # Usage
/// ## Convensional way
/// ```rust,ignore
/// let chain = ConversationalRetriverChainBuilder::new()
///     .llm(llm)
///     .rephrase_question(true)
///     .retriver(RetriverMock {})
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
//  let condense_question_chian = CondenseQuetionGeneratorChain::new(llm.clone_box());
/// let chain = ConversationalRetriverChainBuilder::new()
///     .rephrase_question(true)
///     .combine_documents_chain(Box::new(combine_documents_chain))
///     .condense_question_chian(Box::new(condense_question_chian))
///     .retriver(RetriverMock {})
///     .memory(SimpleMemory::new().into())
///     .build()
///     .expect("Error building ConversationalChain");
/// ```
///
pub struct ConversationalRetriverChainBuilder {
    llm: Option<Box<dyn LLM>>,
    retriver: Option<Box<dyn Retriever>>,
    memory: Option<Arc<Mutex<dyn BaseMemory>>>,
    combine_documents_chain: Option<Box<dyn Chain>>,
    condense_question_chian: Option<Box<dyn Chain>>,
    rephrase_question: bool,
    return_source_documents: bool,
    input_key: String,
    output_key: String,
}
impl ConversationalRetriverChainBuilder {
    pub fn new() -> Self {
        ConversationalRetriverChainBuilder {
            llm: None,
            retriver: None,
            memory: None,
            combine_documents_chain: None,
            condense_question_chian: None,
            rephrase_question: true,
            return_source_documents: true,
            input_key: CONVERSATIONAL_RETRIEVAL_QA_DEFAULT_INPUT_KEY.to_string(),
            output_key: DEFAULT_OUTPUT_KEY.to_string(),
        }
    }

    pub fn retriver<R: Into<Box<dyn Retriever>>>(mut self, retriver: R) -> Self {
        self.retriver = Some(retriver.into());
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
    pub fn combine_documents_chain<C: Into<Box<dyn Chain>>>(
        mut self,
        combine_documents_chain: C,
    ) -> Self {
        self.combine_documents_chain = Some(combine_documents_chain.into());
        self
    }

    ///Chain designed to reformulate the question based on the cat history
    pub fn condense_question_chian<C: Into<Box<dyn Chain>>>(
        mut self,
        condense_question_chian: C,
    ) -> Self {
        self.condense_question_chian = Some(condense_question_chian.into());
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

    pub fn build(mut self) -> Result<ConversationalRetriverChain, ChainError> {
        if let Some(llm) = self.llm {
            let combine_documents_chain = StuffDocument::load_stuff_qa(llm.clone_box());
            let condense_question_chian = CondenseQuetionGeneratorChain::new(llm.clone_box());
            self.combine_documents_chain = Some(Box::new(combine_documents_chain));
            self.condense_question_chian = Some(Box::new(condense_question_chian));
        }

        let retriver = self
            .retriver
            .ok_or_else(|| ChainError::MissingObject("Retriver must be set".into()))?;

        let memory = self
            .memory
            .unwrap_or_else(|| Arc::new(Mutex::new(SimpleMemory::new())));

        let combine_documents_chain = self.combine_documents_chain.ok_or_else(|| {
            ChainError::MissingObject(
                "Combine documents chain must be set or llm must be set".into(),
            )
        })?;
        let condense_question_chian = self.condense_question_chian.ok_or_else(|| {
            ChainError::MissingObject(
                "Condense question chain must be set or llm must be set".into(),
            )
        })?;
        Ok(ConversationalRetriverChain {
            retriver,
            memory,
            combine_documents_chain,
            condense_question_chian,
            rephrase_question: self.rephrase_question,
            return_source_documents: self.return_source_documents,
            input_key: self.input_key,
            output_key: self.output_key,
        })
    }
}
