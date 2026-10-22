//! Offline tests for the refine / map-reduce / retrieval QA chains.
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

use crate::{
    chain::{
        Chain, ChainError, MapReduceDocuments, RefineDocuments, RetrievalQA, StuffDocument,
        DEFAULT_RESULT_KEY, RETRIEVAL_QA_SOURCE_DOCUMENTS_KEY,
    },
    language_models::{llm::LLM, GenerateResult, LLMError, LLMEvent, LLMEventStream, TokenUsage},
    prompt::PromptArgs,
    prompt_args,
    schemas::{Document, Message, Retriever, StreamData},
    test_utils::FakeLLM,
};

fn docs(texts: &[&str]) -> Vec<Document> {
    texts.iter().map(|t| Document::new(*t)).collect()
}

fn prompts(llm: &FakeLLM) -> Vec<String> {
    llm.calls().iter().map(|c| c[0].content.clone()).collect()
}

async fn stream_text(chain: &dyn Chain, input: PromptArgs) -> (String, GenerateResult) {
    let mut text = String::new();
    let mut done = None;
    let mut events = chain.call_stream(input);
    while let Some(event) = events.next().await {
        match event.unwrap() {
            LLMEvent::Text(t) => text.push_str(&t),
            LLMEvent::Done(r) => done = Some(r),
        }
    }
    (text, done.expect("stream ends with Done"))
}

// ---- refine ----

#[tokio::test]
async fn refine_threads_the_answer_through_every_document() {
    let llm = FakeLLM::new(["a1", "a2", "a3"]);
    let chain = RefineDocuments::load_qa(llm.clone());
    let result = chain
        .call(prompt_args! {
            "input_documents" => docs(&["doc one", "doc two", "doc three"]),
            "question" => "q?",
        })
        .await
        .unwrap();

    assert_eq!(result.generation, "a3");
    assert_eq!(
        result.tokens.unwrap().total_tokens,
        6,
        "usage of all 3 calls"
    );

    let p = prompts(&llm);
    assert_eq!(p.len(), 3);
    // First call: initial QA prompt with the first document only.
    assert!(
        p[0].contains("doc one") && p[0].contains("Question:q?"),
        "{}",
        p[0]
    );
    assert!(!p[0].contains("existing answer"));
    // Then each refine call sees the previous answer and the next document.
    assert!(
        p[1].contains("existing answer: a1") && p[1].contains("doc two"),
        "{}",
        p[1]
    );
    assert!(
        p[2].contains("existing answer: a2") && p[2].contains("doc three"),
        "{}",
        p[2]
    );
    assert!(!p[2].contains("doc two"));
}

#[tokio::test]
async fn refine_with_one_document_only_runs_the_initial_chain() {
    let llm = FakeLLM::new(["summary"]);
    let chain = RefineDocuments::load_summarize(llm.clone());
    let out = chain
        .invoke(prompt_args! {"input_documents" => docs(&["only"])})
        .await
        .unwrap();
    assert_eq!(out, "summary");
    assert_eq!(llm.calls().len(), 1);
    assert!(prompts(&llm)[0].contains("Write a concise summary"));
}

#[tokio::test]
async fn refine_rejects_missing_empty_or_malformed_documents() {
    let chain = RefineDocuments::load_summarize(FakeLLM::new(Vec::<String>::new()));
    let err = chain.invoke(prompt_args! {}).await.unwrap_err();
    assert!(matches!(err, ChainError::MissingInputVariable(k) if k == "input_documents"));
    let err = chain
        .invoke(prompt_args! {"input_documents" => Vec::<Document>::new()})
        .await
        .unwrap_err();
    assert!(err.to_string().contains("at least one document"), "{err}");
    let err = chain
        .invoke(prompt_args! {"input_documents" => "nope"})
        .await
        .unwrap_err();
    assert!(matches!(err, ChainError::IncorrectInputVariable { .. }));
}

#[tokio::test]
async fn refine_stops_at_the_first_failing_step() {
    // Only one scripted answer: the second (refine) call fails.
    let llm = FakeLLM::new(["a1"]);
    let chain = RefineDocuments::load_summarize(llm.clone());
    let err = chain
        .invoke(prompt_args! {"input_documents" => docs(&["x", "y", "z"])})
        .await
        .unwrap_err();
    assert!(matches!(err, ChainError::LLMError(_)), "{err}");
    assert_eq!(llm.calls().len(), 2, "no call after the failure");
}

#[tokio::test]
async fn refine_streams_only_the_last_step_with_total_usage() {
    let llm = FakeLLM::new(["first draft", "final answer here"]);
    let chain = RefineDocuments::load_summarize(llm.clone());
    let (text, done) = stream_text(
        &chain,
        prompt_args! {"input_documents" => docs(&["x", "y"])},
    )
    .await;
    assert_eq!(text, "final answer here");
    assert_eq!(done.generation, "final answer here");
    assert_eq!(done.tokens.unwrap().total_tokens, 4);
    assert!(prompts(&llm)[1].contains("first draft"));

    let errors: Vec<_> = chain.call_stream(prompt_args! {}).collect::<Vec<_>>().await;
    assert!(matches!(
        errors.as_slice(),
        [Err(ChainError::MissingInputVariable(_))]
    ));
}

#[tokio::test]
async fn refine_custom_variable_names() {
    let initial = crate::chain::LLMChainBuilder::new()
        .prompt(crate::template_jinja2!("Start: {{doc}}", "doc"))
        .llm(FakeLLM::new(["s"]))
        .build()
        .unwrap();
    let llm = FakeLLM::new(["r"]);
    let refine = crate::chain::LLMChainBuilder::new()
        .prompt(crate::template_jinja2!(
            "Prev {{prev}} + {{doc}}",
            "prev",
            "doc"
        ))
        .llm(llm.clone())
        .build()
        .unwrap();
    let chain = RefineDocuments::new(initial, refine)
        .with_input_key("docs")
        .with_document_variable_name("doc")
        .with_initial_response_name("prev");
    assert_eq!(chain.get_input_keys(), vec!["docs"]);
    let out = chain
        .invoke(prompt_args! {"docs" => docs(&["one", "two"])})
        .await
        .unwrap();
    assert_eq!(out, "r");
    assert_eq!(prompts(&llm), vec!["Prev s + two"]);
}

// ---- map-reduce ----

#[tokio::test]
async fn map_reduce_maps_each_document_then_reduces_in_order() {
    // Map answers come back in document order; the 4th answer is the reduce step.
    let llm = FakeLLM::new(["m1", "m2", "m3", "final"]);
    let chain = MapReduceDocuments::load_qa(llm.clone());
    assert_eq!(chain.get_input_keys(), vec!["input_documents", "question"]);

    let result = chain
        .call(prompt_args! {
            "input_documents" => docs(&["d1", "d2", "d3"]),
            "question" => "q?",
        })
        .await
        .unwrap();
    assert_eq!(result.generation, "final");
    assert_eq!(result.tokens.unwrap().total_tokens, 8, "map + reduce usage");

    let p = prompts(&llm);
    assert_eq!(p.len(), 4);
    for (i, doc) in ["d1", "d2", "d3"].iter().enumerate() {
        assert!(
            p[i].contains(doc) && p[i].contains("Relevant text"),
            "{}",
            p[i]
        );
        assert!(p[i].contains("Question: q?"));
    }
    assert!(p[3].contains("m1\n\nm2\n\nm3"), "{}", p[3]);
    assert!(p[3].contains("QUESTION: q?"));
    assert!(!p[3].contains("d1"), "reduce only sees mapped text");
}

/// LLM whose answer to a prompt is the prompt's document, after a delay that makes
/// later documents finish first. Tracks how many calls run at the same time.
#[derive(Clone, Default)]
struct SlowEcho {
    running: Arc<Mutex<usize>>,
    peak: Arc<Mutex<usize>>,
}

#[async_trait]
impl LLM for SlowEcho {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError> {
        let prompt = messages[0].content.clone();
        {
            let mut running = self.running.lock().unwrap();
            *running += 1;
            let mut peak = self.peak.lock().unwrap();
            *peak = (*peak).max(*running);
        }
        // "doc-N": higher N answers sooner.
        let n: u64 = prompt
            .split("doc-")
            .nth(1)
            .and_then(|s| s.chars().next())
            .and_then(|c| c.to_digit(10))
            .map_or(0, u64::from);
        tokio::time::sleep(Duration::from_millis(50 - n * 10)).await;
        *self.running.lock().unwrap() -= 1;
        Ok(GenerateResult {
            generation: format!("mapped-{n}"),
            tokens: Some(TokenUsage::new(1, 0)),
            ..Default::default()
        })
    }

    async fn stream(
        &self,
        _: &[Message],
    ) -> Result<
        std::pin::Pin<Box<dyn futures::Stream<Item = Result<StreamData, LLMError>> + Send>>,
        LLMError,
    > {
        unimplemented!()
    }

    fn generate_stream<'a>(&'a self, messages: &'a [Message]) -> LLMEventStream<'a> {
        crate::language_models::box_llm_events(async_stream::try_stream! {
            yield LLMEvent::Done(self.generate(messages).await?);
        })
    }
}

#[tokio::test]
async fn map_reduce_runs_maps_concurrently_and_keeps_document_order() {
    let mapper = SlowEcho::default();
    let map_chain = crate::chain::LLMChainBuilder::new()
        .prompt(crate::template_jinja2!("{{context}}", "context"))
        .llm(mapper.clone())
        .build()
        .unwrap();
    let reduce_llm = FakeLLM::new(["done"]);
    let chain =
        MapReduceDocuments::new(map_chain, StuffDocument::load_summarize(reduce_llm.clone()))
            .with_max_concurrency(2);

    let input = docs(&["doc-1", "doc-2", "doc-3", "doc-4"]);
    let result = chain
        .call(prompt_args! {"input_documents" => input})
        .await
        .unwrap();
    assert_eq!(result.generation, "done");
    assert_eq!(
        *mapper.peak.lock().unwrap(),
        2,
        "bounded by max_concurrency"
    );
    let reduce_prompt = &prompts(&reduce_llm)[0];
    assert!(
        reduce_prompt.contains("mapped-1\n\nmapped-2\n\nmapped-3\n\nmapped-4"),
        "{reduce_prompt}"
    );
    // 4 map calls at 1 prompt token each + the reduce call.
    assert_eq!(result.tokens.unwrap().prompt_tokens, 5);
}

#[tokio::test]
async fn map_reduce_passes_metadata_to_the_reduce_chain() {
    struct Spy(Arc<Mutex<Option<PromptArgs>>>);
    #[async_trait]
    impl Chain for Spy {
        async fn call(&self, input: PromptArgs) -> Result<GenerateResult, ChainError> {
            *self.0.lock().unwrap() = Some(input);
            Ok(GenerateResult::default())
        }
    }
    let seen = Arc::new(Mutex::new(None));
    let map_chain = crate::chain::LLMChainBuilder::new()
        .prompt(crate::template_jinja2!(
            "{{context}} {{lang}}",
            "context",
            "lang"
        ))
        .llm(FakeLLM::new(["short"]))
        .build()
        .unwrap();
    let chain =
        MapReduceDocuments::new(map_chain, Spy(seen.clone())).with_reduce_input_key("summaries");
    let doc = Document::new("long text").with_metadata([("source".into(), json!("a.txt"))].into());
    chain
        .call(prompt_args! {"input_documents" => vec![doc], "lang" => "es"})
        .await
        .unwrap();

    let seen = seen.lock().unwrap().clone().unwrap();
    assert_eq!(seen["lang"], "es", "extra inputs reach the reduce chain");
    assert!(!seen.contains_key("input_documents"));
    let mapped: Vec<Document> = serde_json::from_value(seen["summaries"].clone()).unwrap();
    assert_eq!(mapped[0].page_content, "short");
    assert_eq!(mapped[0].metadata["source"], "a.txt");
}

#[tokio::test]
async fn map_reduce_fails_if_any_map_call_fails() {
    let llm = FakeLLM::new(["only one"]);
    let chain = MapReduceDocuments::load_summarize(llm.clone()).with_max_concurrency(1);
    let err = chain
        .invoke(prompt_args! {"input_documents" => docs(&["a", "b"])})
        .await
        .unwrap_err();
    assert!(matches!(err, ChainError::LLMError(_)), "{err}");
    assert_eq!(llm.calls().len(), 2, "reduce never runs");
}

#[tokio::test]
async fn map_reduce_streams_the_reduce_step() {
    let llm = FakeLLM::new(["s1", "s2", "the overall summary"]);
    let chain = MapReduceDocuments::load_summarize(llm.clone()).with_max_concurrency(1);
    let (text, done) = stream_text(
        &chain,
        prompt_args! {"input_documents" => docs(&["a", "b"])},
    )
    .await;
    assert_eq!(text, "the overall summary");
    assert_eq!(done.tokens.unwrap().total_tokens, 6);
    assert!(prompts(&llm)[2].contains("s1\n\ns2"));
}

// ---- stuff summarize / call_stream ----

#[tokio::test]
async fn stuff_summarize_and_call_stream() {
    let llm = FakeLLM::new(["sum one", "sum two"]);
    let chain = StuffDocument::load_summarize(llm.clone());
    let input = || prompt_args! {"input_documents" => docs(&["x", "y"])};
    assert_eq!(chain.invoke(input()).await.unwrap(), "sum one");
    assert!(
        prompts(&llm)[0].contains("\"x\n\ny\""),
        "{}",
        prompts(&llm)[0]
    );

    let (text, done) = stream_text(&chain, input()).await;
    assert_eq!(text, "sum two");
    assert_eq!(done.generation, "sum two");
    let errors: Vec<_> = chain.call_stream(prompt_args! {}).collect().await;
    assert!(matches!(
        errors.as_slice(),
        [Err(ChainError::MissingInputVariable(_))]
    ));
}

// ---- retrieval QA ----

#[derive(Clone)]
struct StaticRetriever {
    docs: Vec<Document>,
    queries: Arc<Mutex<Vec<String>>>,
    fail: bool,
}

impl StaticRetriever {
    fn new(texts: &[&str]) -> Self {
        Self {
            docs: docs(texts),
            queries: Default::default(),
            fail: false,
        }
    }
}

#[async_trait]
impl Retriever for StaticRetriever {
    async fn get_relevant_documents(
        &self,
        query: &str,
    ) -> Result<Vec<Document>, Box<dyn std::error::Error>> {
        self.queries.lock().unwrap().push(query.to_string());
        if self.fail {
            return Err("index down".into());
        }
        Ok(self.docs.clone())
    }
}

#[tokio::test]
async fn retrieval_qa_retrieves_then_answers_with_the_documents() {
    let llm = FakeLLM::new(["Luis is 24"]);
    let retriever = StaticRetriever::new(&["Luis was born in 2002", "It is 2026"]);
    let queries = retriever.queries.clone();
    let qa = RetrievalQA::from_llm(llm.clone(), retriever).with_return_source_documents(true);
    assert_eq!(qa.get_input_keys(), vec!["query"]);

    let out = qa
        .execute(prompt_args! {"query" => "How old is Luis?"})
        .await
        .unwrap();
    assert_eq!(out["output"], "Luis is 24");
    let result: GenerateResult = serde_json::from_value(out[DEFAULT_RESULT_KEY].clone()).unwrap();
    assert_eq!(result.generation, "Luis is 24");
    let sources: Vec<Document> =
        serde_json::from_value(out[RETRIEVAL_QA_SOURCE_DOCUMENTS_KEY].clone()).unwrap();
    assert_eq!(sources.len(), 2);

    assert_eq!(*queries.lock().unwrap(), vec!["How old is Luis?"]);
    let prompt = &prompts(&llm)[0];
    assert!(
        prompt.contains("Luis was born in 2002\n\nIt is 2026"),
        "{prompt}"
    );
    assert!(prompt.contains("Question:How old is Luis?"));
}

#[tokio::test]
async fn retrieval_qa_without_sources_and_with_custom_keys() {
    let qa = RetrievalQA::from_llm(FakeLLM::new(["a"]), StaticRetriever::new(&["d"]))
        .with_input_key("question")
        .with_output_key("answer");
    let out = qa.execute(prompt_args! {"question" => "q"}).await.unwrap();
    assert_eq!(out["answer"], "a");
    assert!(!out.contains_key(RETRIEVAL_QA_SOURCE_DOCUMENTS_KEY));
    assert_eq!(qa.get_output_keys(), vec!["answer", DEFAULT_RESULT_KEY]);
}

#[tokio::test]
async fn retrieval_qa_errors() {
    let qa = RetrievalQA::from_llm(FakeLLM::new(["a"]), StaticRetriever::new(&["d"]));
    let err = qa.invoke(prompt_args! {}).await.unwrap_err();
    assert!(matches!(err, ChainError::MissingInputVariable(k) if k == "query"));

    let mut failing = StaticRetriever::new(&[]);
    failing.fail = true;
    let llm = FakeLLM::new(["a"]);
    let qa = RetrievalQA::from_llm(llm.clone(), failing);
    let err = qa.invoke(prompt_args! {"query" => "q"}).await.unwrap_err();
    assert!(
        matches!(err, ChainError::RetrieverError(ref m) if m == "index down"),
        "{err}"
    );
    assert!(llm.calls().is_empty());
}

#[tokio::test]
async fn retrieval_qa_with_refine_and_streaming() {
    let llm = FakeLLM::new(["draft", "better answer"]);
    let qa = RetrievalQA::new(
        StaticRetriever::new(&["d1", "d2"]),
        RefineDocuments::load_qa(llm.clone()),
    );
    let (text, done) = stream_text(&qa, prompt_args! {"query" => "q?"}).await;
    assert_eq!(text, "better answer");
    assert_eq!(done.tokens.unwrap().total_tokens, 4);
    assert!(prompts(&llm)[1].contains("The original question is as follows: q?"));
}
