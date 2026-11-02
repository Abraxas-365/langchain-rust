use std::{
    error::Error,
    pin::Pin,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use langchain_rust::{
    chain::{
        Chain, CondenseQuestionGeneratorChain, ConversationalRetrieverChain,
        ConversationalRetrieverChainBuilder, StuffDocument,
    },
    language_models::{llm::LLM, GenerateResult, LLMError, LLMEvent, TokenUsage},
    llm::{CachedLLM, InMemoryLLMCache},
    memory::SimpleMemory,
    prompt_args,
    schemas::{BaseMemory, Document, Message, Retriever, StreamData},
};
use tokio::sync::Mutex;

struct Documents;

#[async_trait]
impl Retriever for Documents {
    async fn get_relevant_documents(&self, _: &str) -> Result<Vec<Document>, Box<dyn Error>> {
        Ok(vec![Document::new("fixture document")])
    }
}

#[derive(Clone)]
struct Model {
    text: &'static str,
    usage: Option<TokenUsage>,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl LLM for Model {
    async fn generate(&self, _: &[Message]) -> Result<GenerateResult, LLMError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(GenerateResult {
            generation: self.text.into(),
            tokens: self.usage.clone(),
            ..Default::default()
        })
    }

    async fn stream(
        &self,
        _: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        Err(LLMError::OtherError("fixture does not stream".into()))
    }
}

fn model(text: &'static str, usage: Option<TokenUsage>) -> Model {
    Model {
        text,
        usage,
        calls: Arc::new(AtomicUsize::new(0)),
    }
}

async fn memory(with_history: bool) -> Arc<Mutex<dyn BaseMemory>> {
    let memory: Arc<Mutex<dyn BaseMemory>> = Arc::new(Mutex::new(SimpleMemory::new()));
    if with_history {
        let mut guard = memory.lock().await;
        guard.add_user_message(&"prior user").await.unwrap();
        guard.add_ai_message(&"prior answer").await.unwrap();
    }
    memory
}

async fn result(chain: &ConversationalRetrieverChain, mode: u8) -> GenerateResult {
    let inputs = prompt_args! { "question" => "new question" };
    match mode {
        0 => chain.call(inputs).await.unwrap(),
        1 => {
            let output = chain.execute(inputs).await.unwrap();
            assert_eq!(output["output"], "answer");
            assert_eq!(output["source_documents"].as_array().unwrap().len(), 1);
            serde_json::from_value(output["generate_result"].clone()).unwrap()
        }
        _ => {
            let mut stream = chain.call_stream(inputs);
            let mut final_result = None;
            while let Some(event) = stream.next().await {
                if let LLMEvent::Done(output) = event.unwrap() {
                    assert!(final_result.is_none());
                    final_result = Some(output);
                }
            }
            final_result.unwrap()
        }
    }
}

#[tokio::test]
async fn answer_cache_hits_keep_new_condensation_usage() {
    for mode in 0..3 {
        let condense_usage = TokenUsage::new(10, 2).with_cache_tokens(2, 1);
        let answer_usage = TokenUsage::new(3, 4).with_cache_tokens(1, 2);
        let condense = model("standalone question", Some(condense_usage.clone()));
        let answer = model("answer", Some(answer_usage.clone()));
        let condense_calls = condense.calls.clone();
        let answer_calls = answer.calls.clone();
        let cache = Arc::new(InMemoryLLMCache::new());
        let chain = ConversationalRetrieverChainBuilder::new()
            .retriever(Documents)
            .memory(memory(true).await)
            .condense_question_chain(CondenseQuestionGeneratorChain::new(condense))
            .combine_documents_chain(StuffDocument::load_stuff_qa(CachedLLM::new(answer, cache)))
            .build()
            .unwrap();
        let first = result(&chain, 0).await;
        assert_eq!(first.tokens, Some(condense_usage.sum(&answer_usage)));
        let cached = result(&chain, mode).await;
        assert_eq!(cached.generation, first.generation);
        assert_eq!(cached.tokens, Some(condense_usage));
        assert_eq!(condense_calls.load(Ordering::SeqCst), 2);
        assert_eq!(answer_calls.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn optional_usage_and_disabled_condensation_are_preserved() {
    for with_history in [false, true] {
        for rephrase in [false, true] {
            for condense_usage in [None, Some(TokenUsage::new(10, 2).with_cache_tokens(2, 1))] {
                for answer_usage in [None, Some(TokenUsage::new(3, 4).with_cache_tokens(1, 2))] {
                    for mode in 0..3 {
                        let condense = model("standalone question", condense_usage.clone());
                        let calls = condense.calls.clone();
                        let chain = ConversationalRetrieverChainBuilder::new()
                            .retriever(Documents)
                            .memory(memory(with_history).await)
                            .rephrase_question(rephrase)
                            .condense_question_chain(CondenseQuestionGeneratorChain::new(condense))
                            .combine_documents_chain(StuffDocument::load_stuff_qa(model(
                                "answer",
                                answer_usage.clone(),
                            )))
                            .build()
                            .unwrap();
                        let actual = result(&chain, mode).await;
                        let mut expected = answer_usage.clone();
                        if with_history && rephrase {
                            if let Some(usage) = &condense_usage {
                                expected = Some(match expected {
                                    Some(answer) => usage.sum(&answer),
                                    None => usage.clone(),
                                });
                            }
                        }
                        assert_eq!(actual.tokens, expected);
                        assert_eq!(actual.generation, "answer");
                        assert_eq!(
                            calls.load(Ordering::SeqCst),
                            usize::from(with_history && rephrase)
                        );
                    }
                }
            }
        }
    }
}
