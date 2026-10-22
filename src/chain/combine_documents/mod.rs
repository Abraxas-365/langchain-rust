//! Chains that combine many documents into one answer, beyond
//! [`StuffDocument`](crate::chain::StuffDocument): refine, map-reduce, map-rerank,
//! summarization, and [`RetrievalQA`] on top of a retriever.
mod map_reduce;
mod map_rerank;
mod prompts;
mod refine;
mod retrieval_qa;

pub use map_reduce::MapReduceDocuments;
pub use map_rerank::{
    MapRerankDocuments, RankedAnswer, MAP_RERANK_INTERMEDIATE_STEPS_KEY, MAP_RERANK_SCORE_KEY,
};
pub use refine::RefineDocuments;
pub use retrieval_qa::{RetrievalQA, RETRIEVAL_QA_SOURCE_DOCUMENTS_KEY};

pub(crate) use prompts::STUFF_QA_TEMPLATE;

use crate::{
    chain::{LLMChain, LLMChainBuilder, StuffDocument},
    language_models::llm::LLM,
    prompt::{PromptTemplate, TemplateFormat},
};

fn llm_chain(llm: Box<dyn LLM>, template: &str, variables: &[&str]) -> LLMChain {
    let prompt = PromptTemplate::new(
        template.to_string(),
        variables.iter().map(|v| v.to_string()).collect(),
        TemplateFormat::Jinja2,
    );
    LLMChainBuilder::new()
        .prompt(prompt)
        .llm(llm)
        .build()
        .expect("prompt and llm are set")
}

impl StuffDocument {
    /// Summarizes all documents in one prompt. Input: `input_documents`.
    pub fn load_summarize<L: Into<Box<dyn LLM>>>(llm: L) -> Self {
        StuffDocument::new(llm_chain(
            llm.into(),
            prompts::SUMMARIZE_TEMPLATE,
            &["context"],
        ))
    }
}

impl RefineDocuments {
    /// Answers `question` from the first document, then refines the answer with each of the
    /// others. Inputs: `input_documents`, `question`.
    pub fn load_qa<L: Into<Box<dyn LLM>>>(llm: L) -> Self {
        let llm = llm.into();
        RefineDocuments::new(
            llm_chain(
                llm.clone_box(),
                prompts::STUFF_QA_TEMPLATE,
                &["context", "question"],
            ),
            llm_chain(
                llm,
                prompts::REFINE_QA_TEMPLATE,
                &["question", "existing_answer", "context"],
            ),
        )
    }

    /// Summarizes the first document, then refines the summary with each of the others.
    /// Input: `input_documents`.
    pub fn load_summarize<L: Into<Box<dyn LLM>>>(llm: L) -> Self {
        let llm = llm.into();
        RefineDocuments::new(
            llm_chain(llm.clone_box(), prompts::SUMMARIZE_TEMPLATE, &["context"]),
            llm_chain(
                llm,
                prompts::REFINE_SUMMARIZE_TEMPLATE,
                &["existing_answer", "context"],
            ),
        )
    }
}

impl MapReduceDocuments {
    /// Extracts the passages relevant to `question` from each document, then answers from
    /// those. Inputs: `input_documents`, `question`.
    pub fn load_qa<L: Into<Box<dyn LLM>>>(llm: L) -> Self {
        let llm = llm.into();
        MapReduceDocuments::new(
            llm_chain(
                llm.clone_box(),
                prompts::MAP_REDUCE_MAP_QA_TEMPLATE,
                &["context", "question"],
            ),
            StuffDocument::new(llm_chain(
                llm,
                prompts::MAP_REDUCE_COMBINE_QA_TEMPLATE,
                &["context", "question"],
            )),
        )
    }

    /// Summarizes each document, then summarizes the summaries. Input: `input_documents`.
    pub fn load_summarize<L: Into<Box<dyn LLM>>>(llm: L) -> Self {
        let llm = llm.into();
        MapReduceDocuments::new(
            llm_chain(llm.clone_box(), prompts::SUMMARIZE_TEMPLATE, &["context"]),
            StuffDocument::load_summarize(llm),
        )
    }
}

impl MapRerankDocuments {
    /// Answers `question` from each document with a self-reported score and returns the
    /// best-scored answer. Inputs: `input_documents`, `question`.
    pub fn load_qa<L: Into<Box<dyn LLM>>>(llm: L) -> Self {
        MapRerankDocuments::new(llm_chain(
            llm.into(),
            prompts::MAP_RERANK_QA_TEMPLATE,
            &["context", "question"],
        ))
    }
}

#[cfg(test)]
mod tests;
