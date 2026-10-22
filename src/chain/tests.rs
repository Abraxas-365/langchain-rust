//! Offline tests for chains, driven by a scripted fake LLM.
use std::sync::Arc;

use futures::StreamExt;
use tokio::sync::Mutex;

use crate::{
    chain::{
        builder::ConversationalChainBuilder, options::ChainCallOptions, Chain, ChainError,
        LLMChainBuilder, StuffDocument, DEFAULT_RESULT_KEY,
    },
    language_models::GenerateResult,
    memory::WindowBufferMemory,
    output_parsers::SimpleParser,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::{memory::BaseMemory, Document, MessageType},
    sequential_chain, template_fstring,
    test_utils::FakeLLM,
};

fn greet_chain(llm: FakeLLM) -> crate::chain::LLMChain {
    LLMChainBuilder::new()
        .prompt(HumanMessagePromptTemplate::new(template_fstring!(
            "Hello {name}!",
            "name"
        )))
        .llm(llm)
        .build()
        .unwrap()
}

#[tokio::test]
async fn llm_chain_formats_prompt_and_parses_output() {
    let llm = FakeLLM::new(["  hi luis  "]);
    let chain = LLMChainBuilder::new()
        .prompt(HumanMessagePromptTemplate::new(template_fstring!(
            "Hello {name}!",
            "name"
        )))
        .llm(llm.clone())
        .output_parser(SimpleParser::new().with_trim(true))
        .output_key("greeting")
        .options(ChainCallOptions::new().with_max_tokens(5))
        .build()
        .unwrap();

    assert_eq!(chain.get_input_keys(), vec!["name".to_string()]);
    assert_eq!(chain.get_output_keys(), vec!["greeting".to_string()]);

    let result = chain.call(prompt_args! {"name" => "luis"}).await.unwrap();
    assert_eq!(result.generation, "hi luis");
    assert_eq!(result.tokens.unwrap().total_tokens, 2);

    let calls = llm.calls();
    assert_eq!(calls[0].len(), 1);
    assert_eq!(calls[0][0].content, "Hello luis!");
    assert_eq!(calls[0][0].message_type, MessageType::HumanMessage);
    assert_eq!(llm.options.lock().unwrap()[0].max_tokens, Some(5));
}

#[tokio::test]
async fn llm_chain_invoke_skips_output_parser() {
    let llm = FakeLLM::new(["  raw  "]);
    let chain = LLMChainBuilder::new()
        .prompt(HumanMessagePromptTemplate::new(template_fstring!(
            "{x}", "x"
        )))
        .llm(llm)
        .output_parser(SimpleParser::new().with_trim(true))
        .build()
        .unwrap();
    // invoke() returns the raw generation (documented current behaviour)
    assert_eq!(
        chain.invoke(prompt_args! {"x" => "1"}).await.unwrap(),
        "  raw  "
    );
}

#[tokio::test]
async fn llm_chain_missing_variable_and_builder_errors() {
    let chain = greet_chain(FakeLLM::new(["x"]));
    let err = chain.call(prompt_args! {"other" => "v"}).await.unwrap_err();
    assert!(matches!(err, ChainError::PromptError(_)), "{err:?}");

    assert!(matches!(
        LLMChainBuilder::new().llm(FakeLLM::default()).build(),
        Err(ChainError::MissingObject(_))
    ));
    assert!(matches!(
        LLMChainBuilder::new()
            .prompt(HumanMessagePromptTemplate::new(template_fstring!("x",)))
            .build(),
        Err(ChainError::MissingObject(_))
    ));
}

#[tokio::test]
async fn llm_chain_propagates_llm_errors() {
    let chain = greet_chain(FakeLLM::default());
    let err = chain.call(prompt_args! {"name" => "a"}).await.unwrap_err();
    assert!(matches!(err, ChainError::LLMError(_)), "{err:?}");
}

#[tokio::test]
async fn llm_chain_execute_and_stream() {
    let chain = greet_chain(FakeLLM::new(["one", "two words here"]));
    let out = chain.execute(prompt_args! {"name" => "a"}).await.unwrap();
    assert_eq!(out["output"], "one");
    let result: GenerateResult = serde_json::from_value(out[DEFAULT_RESULT_KEY].clone()).unwrap();
    assert_eq!(result.generation, "one");

    let chunks: Vec<String> = chain
        .stream(prompt_args! {"name" => "a"})
        .await
        .unwrap()
        .map(|r| r.unwrap().content)
        .collect()
        .await;
    assert_eq!(chunks, vec!["two ", "words ", "here"]);
}

#[tokio::test]
async fn conversational_chain_keeps_history() {
    let llm = FakeLLM::new(["Nice to meet you", "You are from Peru"]);
    let memory = Arc::new(Mutex::new(WindowBufferMemory::new(10)));
    let chain = ConversationalChainBuilder::new()
        .llm(llm.clone())
        .memory(memory.clone())
        .build()
        .unwrap();

    let input = chain.prompt_builder().input("I'm from Peru").build();
    assert_eq!(chain.invoke(input).await.unwrap(), "Nice to meet you");
    let r = chain
        .invoke(prompt_args! {"input" => "Where am I from?"})
        .await
        .unwrap();
    assert_eq!(r, "You are from Peru");

    let second_prompt = &llm.calls()[1][0].content;
    assert!(second_prompt.contains("human: I'm from Peru"));
    assert!(second_prompt.contains("ai: Nice to meet you"));
    assert!(second_prompt.contains("Human: Where am I from?"));
    assert_eq!(memory.lock().await.messages().await.unwrap().len(), 4);

    let err = chain
        .invoke(prompt_args! {"wrong" => "x"})
        .await
        .unwrap_err();
    assert!(matches!(err, ChainError::MissingInputVariable(k) if k == "input"));
}

#[tokio::test]
async fn conversational_chain_stream_updates_memory() {
    let llm = FakeLLM::new(["streamed answer"]);
    let chain = ConversationalChainBuilder::new().llm(llm).build().unwrap();
    let text: String = chain
        .stream(prompt_args! {"input" => "hi"})
        .await
        .unwrap()
        .map(|r| r.unwrap().content)
        .collect()
        .await;
    assert_eq!(text, "streamed answer");
    let messages = chain.memory.lock().await.messages().await.unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1].content, "streamed answer");
}

#[tokio::test]
async fn sequential_chain_pipes_outputs_and_sums_tokens() {
    let first = LLMChainBuilder::new()
        .prompt(HumanMessagePromptTemplate::new(template_fstring!(
            "Name a dish from {country}",
            "country"
        )))
        .llm(FakeLLM::new(["ceviche"]))
        .output_key("dish")
        .build()
        .unwrap();
    let second_llm = FakeLLM::new(["fish and lime"]);
    let second = LLMChainBuilder::new()
        .prompt(HumanMessagePromptTemplate::new(template_fstring!(
            "Ingredients of {dish}",
            "dish"
        )))
        .llm(second_llm.clone())
        .output_key("ingredients")
        .build()
        .unwrap();

    let chain = sequential_chain!(first, second);
    let out = chain
        .execute(prompt_args! {"country" => "Peru"})
        .await
        .unwrap();
    assert_eq!(out["dish"], "ceviche");
    assert_eq!(out["ingredients"], "fish and lime");
    let result: GenerateResult = serde_json::from_value(out[DEFAULT_RESULT_KEY].clone()).unwrap();
    assert_eq!(result.generation, "fish and lime");
    assert_eq!(result.tokens.unwrap().total_tokens, 4);
    assert_eq!(second_llm.calls()[0][0].content, "Ingredients of ceviche");
}

#[tokio::test]
async fn stuff_document_qa_joins_documents() {
    let llm = FakeLLM::new(["24 and nvim"]);
    let chain = StuffDocument::load_stuff_qa(llm.clone());
    let docs = [Document::new("Luis is 24"), Document::new("Luis uses nvim")];
    let input = chain
        .qa_prompt_builder()
        .documents(&docs)
        .question("How old is Luis and which editor?")
        .build();
    assert_eq!(chain.invoke(input).await.unwrap(), "24 and nvim");

    let prompt = &llm.calls()[0][0].content;
    assert!(prompt.contains("Luis is 24\n\nLuis uses nvim"), "{prompt}");
    assert!(prompt.contains("Question:How old is Luis and which editor?"));

    let err = chain
        .invoke(prompt_args! {"question" => "q"})
        .await
        .unwrap_err();
    assert!(matches!(err, ChainError::MissingInputVariable(_)));
    let err = chain
        .invoke(prompt_args! {"input_documents" => "not docs", "question" => "q"})
        .await
        .unwrap_err();
    assert!(matches!(err, ChainError::IncorrectInputVariable { .. }));
}

struct StaticRetriever {
    docs: Vec<Document>,
    queries: Arc<std::sync::Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl crate::schemas::Retriever for StaticRetriever {
    async fn get_relevant_documents(
        &self,
        query: &str,
    ) -> Result<Vec<Document>, Box<dyn std::error::Error>> {
        self.queries.lock().unwrap().push(query.to_string());
        Ok(self.docs.clone())
    }
}

#[tokio::test]
async fn conversational_retriever_chain_rephrases_with_history() {
    use crate::chain::ConversationalRetrieverChainBuilder;

    // 1st turn: no history -> no rephrase, only the QA call.
    // 2nd turn: rephrase call, then QA call.
    let llm = FakeLLM::new(["Luis is 24", "How old is Luis's brother?", "He is 30"]);
    let queries = Arc::new(std::sync::Mutex::new(Vec::new()));
    let chain = ConversationalRetrieverChainBuilder::new()
        .llm(llm.clone())
        .retriever(StaticRetriever {
            docs: vec![Document::new("Luis is 24. His brother is 30.")],
            queries: queries.clone(),
        })
        .rephrase_question(true)
        .return_source_documents(true)
        .build()
        .unwrap();

    let out = chain
        .execute(prompt_args! {"question" => "How old is Luis?"})
        .await
        .unwrap();
    assert_eq!(out["output"], "Luis is 24");
    assert_eq!(
        out["source_documents"][0]["page_content"],
        "Luis is 24. His brother is 30."
    );

    let out = chain
        .execute(prompt_args! {"question" => "And his brother?"})
        .await
        .unwrap();
    assert_eq!(out["output"], "He is 30");
    assert_eq!(out["generated_question"], "How old is Luis's brother?");

    assert_eq!(
        *queries.lock().unwrap(),
        vec!["How old is Luis?", "How old is Luis's brother?"]
    );
    let calls = llm.calls();
    assert_eq!(calls.len(), 3);
    assert!(calls[1][0]
        .content
        .contains("Follow Up Input: And his brother?"));
    // user input is stored unquoted in memory
    let messages = chain.memory.lock().await.messages().await.unwrap();
    assert_eq!(messages[0].content, "How old is Luis?");
    assert_eq!(messages.len(), 4);
}

#[tokio::test]
async fn conversational_retriever_chain_requires_retriever_and_llm() {
    use crate::chain::ConversationalRetrieverChainBuilder;
    assert!(matches!(
        ConversationalRetrieverChainBuilder::new()
            .llm(FakeLLM::default())
            .build(),
        Err(ChainError::MissingObject(_))
    ));
    assert!(matches!(
        ConversationalRetrieverChainBuilder::new()
            .retriever(StaticRetriever {
                docs: vec![],
                queries: Default::default()
            })
            .build(),
        Err(ChainError::MissingObject(_))
    ));
}
