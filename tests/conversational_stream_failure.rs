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
        conversational::builder::ConversationalChainBuilder, Chain,
        ConversationalRetrieverChainBuilder,
    },
    language_models::{llm::LLM, GenerateResult, LLMError},
    memory::SimpleMemory,
    prompt_args,
    schemas::{BaseMemory, Document, Message, Retriever, StreamData},
};
use serde_json::json;
use tokio::sync::Mutex;

struct Documents;

#[async_trait]
impl Retriever for Documents {
    async fn get_relevant_documents(&self, _: &str) -> Result<Vec<Document>, Box<dyn Error>> {
        Ok(vec![Document::new("fixture")])
    }
}

#[derive(Clone, Copy)]
enum Outcome {
    Failed,
    FailedAtEof,
    ImmediateFailure,
    Success,
    EmptySuccess,
}

#[derive(Clone)]
struct ScriptedLLM {
    outcome: Outcome,
    polls: Arc<AtomicUsize>,
}

#[async_trait]
impl LLM for ScriptedLLM {
    async fn generate(&self, _: &[Message]) -> Result<GenerateResult, LLMError> {
        match self.outcome {
            Outcome::Failed | Outcome::FailedAtEof | Outcome::ImmediateFailure => {
                Err(LLMError::OtherError("fixture failure".into()))
            }
            _ => Ok(GenerateResult {
                generation: "complete".into(),
                ..Default::default()
            }),
        }
    }

    async fn stream(
        &self,
        _: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let text = |value: &str| Ok(StreamData::new(json!(value), None, value));
        let error = || Err(LLMError::OtherError("fixture failure".into()));
        let events = match self.outcome {
            Outcome::Failed => vec![text("partial"), error(), text("after error")],
            Outcome::FailedAtEof => vec![text("partial"), error()],
            Outcome::ImmediateFailure => vec![error()],
            Outcome::Success => vec![text("complete")],
            Outcome::EmptySuccess => vec![],
        };
        let polls = self.polls.clone();
        Ok(Box::pin(futures::stream::iter(events).inspect(move |_| {
            polls.fetch_add(1, Ordering::SeqCst);
        })))
    }
}

async fn fixture(
    retrieval: bool,
    outcome: Outcome,
) -> (Box<dyn Chain>, Arc<Mutex<dyn BaseMemory>>, Arc<AtomicUsize>) {
    let memory: Arc<Mutex<dyn BaseMemory>> = Arc::new(Mutex::new(SimpleMemory::new()));
    for message in [
        Message::new_human_message("prior user").with_images(vec!["data:image/png;base64,AAAA"]),
        Message::new_ai_message("checking").with_tool_calls(json!([{"id":"prior_call","type":"function","function":{"name":"lookup","arguments":"{}"}}])),
        Message::new_tool_message("prior result", "prior_call"),
        Message::new_ai_message("prior answer"),
    ] {
        memory.lock().await.add_message(message).await.unwrap();
    }
    let polls = Arc::new(AtomicUsize::new(0));
    let llm = ScriptedLLM {
        outcome,
        polls: polls.clone(),
    };
    let chain: Box<dyn Chain> = if retrieval {
        Box::new(
            ConversationalRetrieverChainBuilder::new()
                .llm(llm)
                .retriever(Documents)
                .rephrase_question(false)
                .memory(memory.clone())
                .build()
                .unwrap(),
        )
    } else {
        Box::new(
            ConversationalChainBuilder::new()
                .llm(llm)
                .memory(memory.clone())
                .build()
                .unwrap(),
        )
    };
    (chain, memory, polls)
}

fn inputs(retrieval: bool) -> langchain_rust::prompt::PromptArgs {
    if retrieval {
        prompt_args! { "question" => "new question" }
    } else {
        prompt_args! { "input" => "new question", "images" => json!(["data:image/png;base64,BBBB"]) }
    }
}

#[tokio::test]
async fn provider_error_is_terminal_and_does_not_commit_a_turn() {
    for retrieval in [false, true] {
        for outcome in [
            Outcome::Failed,
            Outcome::FailedAtEof,
            Outcome::ImmediateFailure,
        ] {
            let (chain, memory, polls) = fixture(retrieval, outcome).await;
            let before = json!(memory.lock().await.messages().await.unwrap());
            let mut stream = chain.stream(inputs(retrieval)).await.unwrap();
            let mut texts = vec![];
            let mut errors = vec![];
            while let Some(result) = stream.next().await {
                match result {
                    Ok(data) => texts.push(data.content),
                    Err(error) => errors.push(error.to_string()),
                }
            }
            assert_eq!(errors.len(), 1);
            assert!(errors[0].contains("fixture failure"));
            let expected = if matches!(outcome, Outcome::Failed | Outcome::FailedAtEof) {
                vec!["partial"]
            } else {
                vec![]
            };
            assert_eq!(texts, expected);
            assert_eq!(polls.load(Ordering::SeqCst), expected.len() + 1);
            assert_eq!(json!(memory.lock().await.messages().await.unwrap()), before);
        }
    }
}

#[tokio::test]
async fn successful_streams_commit_even_when_empty() {
    for retrieval in [false, true] {
        for outcome in [Outcome::Success, Outcome::EmptySuccess] {
            let (chain, memory, _) = fixture(retrieval, outcome).await;
            let before = memory.lock().await.messages().await.unwrap();
            let mut stream = chain.stream(inputs(retrieval)).await.unwrap();
            let mut text = String::new();
            while let Some(result) = stream.next().await {
                text.push_str(&result.unwrap().content);
            }
            let after = memory.lock().await.messages().await.unwrap();
            assert_eq!(after.len(), before.len() + 2);
            assert_eq!(json!(&after[..before.len()]), json!(before));
            assert_eq!(after[4].content, "new question");
            assert_eq!(after[5].content, text);
            if !retrieval {
                assert_eq!(
                    after[4].images.as_ref().unwrap()[0].image_url,
                    "data:image/png;base64,BBBB"
                );
            }
        }
    }
}

#[tokio::test]
async fn dropping_partial_streams_preserves_history() {
    for retrieval in [false, true] {
        let (chain, memory, polls) = fixture(retrieval, Outcome::Failed).await;
        let before = json!(memory.lock().await.messages().await.unwrap());
        let mut stream = chain.stream(inputs(retrieval)).await.unwrap();
        assert_eq!(stream.next().await.unwrap().unwrap().content, "partial");
        drop(stream);
        assert_eq!(polls.load(Ordering::SeqCst), 1);
        assert_eq!(json!(memory.lock().await.messages().await.unwrap()), before);
    }
}
