//! A chain that runs your own function over its inputs.

use std::{collections::HashMap, future::Future, sync::Arc};

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde_json::{json, Value};

use crate::{
    chain::{Chain, ChainError, DEFAULT_RESULT_KEY},
    language_models::GenerateResult,
    observability::traced,
    prompt::PromptArgs,
    schemas::messages::value_to_text,
};

type TransformFn = Arc<
    dyn Fn(PromptArgs) -> BoxFuture<'static, Result<HashMap<String, Value>, ChainError>>
        + Send
        + Sync,
>;

/// A chain that runs a function over its inputs, e.g. to clean or reshape text between
/// two LLM chains in a [`SequentialChain`](super::SequentialChain).
///
/// ```rust
/// use langchain_rust::{chain::{Chain, TransformChain}, prompt_args};
/// use serde_json::json;
/// use std::collections::HashMap;
///
/// let first_line = TransformChain::new(["text"], ["first_line"], |inputs| async move {
///     let text = inputs["text"].as_str().unwrap_or_default().to_string();
///     let line = text.lines().next().unwrap_or_default().to_string();
///     Ok(HashMap::from([("first_line".to_string(), json!(line))]))
/// });
/// # tokio_test::block_on(async {
/// let out = first_line.invoke(prompt_args! {"text" => "one\ntwo"}).await.unwrap();
/// assert_eq!(out, "one");
/// # });
/// ```
///
/// The function gets every input, not only the declared ones. Before it runs, each
/// input key must be present; after, each output key must be in what it returned (extra
/// keys are kept). [`execute`](Chain::execute) returns the function's outputs, and
/// [`call`](Chain::call) / [`invoke`](Chain::invoke) return the first output key's value
/// as text. In a `SequentialChain` only that first output is passed on.
///
/// Like LangChain's and langchaingo's `TransformChain`.
#[derive(Clone)]
pub struct TransformChain {
    input_keys: Vec<String>,
    output_keys: Vec<String>,
    transform: TransformFn,
}

impl TransformChain {
    /// `transform` takes the chain's inputs and returns its outputs, which must include
    /// every key in `output_keys`.
    pub fn new<I, O, F, Fut>(input_keys: I, output_keys: O, transform: F) -> Self
    where
        I: IntoIterator,
        I::Item: Into<String>,
        O: IntoIterator,
        O::Item: Into<String>,
        F: Fn(PromptArgs) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<HashMap<String, Value>, ChainError>> + Send + 'static,
    {
        Self {
            input_keys: input_keys.into_iter().map(Into::into).collect(),
            output_keys: output_keys.into_iter().map(Into::into).collect(),
            transform: Arc::new(move |inputs| Box::pin(transform(inputs))),
        }
    }

    async fn run(
        &self,
        inputs: PromptArgs,
    ) -> Result<(HashMap<String, Value>, GenerateResult), ChainError> {
        if let Some(missing) = self.input_keys.iter().find(|k| !inputs.contains_key(*k)) {
            return Err(ChainError::MissingInputVariable(missing.clone()));
        }
        let outputs = (self.transform)(inputs).await?;
        if let Some(missing) = self.output_keys.iter().find(|k| !outputs.contains_key(*k)) {
            return Err(ChainError::OtherError(format!(
                "transform did not return output key `{missing}`"
            )));
        }
        let generation = self
            .output_keys
            .first()
            .map(|key| value_to_text(&outputs[key]))
            .unwrap_or_default();
        Ok((
            outputs,
            GenerateResult {
                generation,
                ..Default::default()
            },
        ))
    }

    async fn run_traced(
        &self,
        inputs: PromptArgs,
    ) -> Result<(HashMap<String, Value>, GenerateResult), ChainError> {
        let span = tracing::info_span!(
            "chain.call",
            chain = "transform",
            error = tracing::field::Empty
        );
        traced(span, self.run(inputs), |_, _| {}).await
    }
}

#[async_trait]
impl Chain for TransformChain {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        Ok(self.run_traced(input_variables).await?.1)
    }

    async fn execute(
        &self,
        input_variables: PromptArgs,
    ) -> Result<HashMap<String, Value>, ChainError> {
        let (mut outputs, result) = self.run_traced(input_variables).await?;
        outputs.insert(DEFAULT_RESULT_KEY.to_string(), json!(result));
        Ok(outputs)
    }

    fn get_input_keys(&self) -> Vec<String> {
        self.input_keys.clone()
    }

    fn get_output_keys(&self) -> Vec<String> {
        self.output_keys.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use futures::StreamExt;

    use super::*;
    use crate::{
        chain::LLMChainBuilder, language_models::LLMEvent, prompt::HumanMessagePromptTemplate,
        prompt_args, sequential_chain, template_fstring, test_utils::FakeLLM,
    };

    fn out(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    fn upper() -> TransformChain {
        TransformChain::new(["text"], ["upper", "len"], |inputs| async move {
            let text = value_to_text(&inputs["text"]);
            Ok(out(&[
                ("upper", json!(text.to_uppercase())),
                ("len", json!(text.len())),
            ]))
        })
    }

    #[tokio::test]
    async fn runs_the_function_and_returns_its_outputs() {
        let chain = upper();
        assert_eq!(chain.get_input_keys(), vec!["text"]);
        assert_eq!(chain.get_output_keys(), vec!["upper", "len"]);
        assert_eq!(
            chain.invoke(prompt_args! {"text" => "abc"}).await.unwrap(),
            "ABC"
        );
        let result = chain.call(prompt_args! {"text" => "abc"}).await.unwrap();
        assert_eq!(result.generation, "ABC");
        assert!(result.tokens.is_none());

        let executed = chain.execute(prompt_args! {"text" => "hey"}).await.unwrap();
        assert_eq!(executed.len(), 3, "{executed:?}");
        assert_eq!(executed["upper"], json!("HEY"));
        assert_eq!(executed["len"], json!(3));
        assert_eq!(executed[DEFAULT_RESULT_KEY]["generation"], json!("HEY"));
    }

    #[tokio::test]
    async fn gets_every_input_and_keeps_extra_outputs() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorder = seen.clone();
        let chain = TransformChain::new(["a"], ["n"], move |inputs| {
            let mut keys: Vec<String> = inputs.keys().cloned().collect();
            keys.sort();
            recorder.lock().unwrap().push(keys);
            async move { Ok(out(&[("n", json!(7)), ("extra", json!([1, 2]))])) }
        });
        let executed = chain
            .execute(prompt_args! {"a" => 1, "b" => 2})
            .await
            .unwrap();
        assert_eq!(seen.lock().unwrap().clone(), vec![vec!["a", "b"]]);
        assert_eq!(executed["extra"], json!([1, 2]));
        // A non-string first output is rendered as JSON text.
        assert_eq!(chain.invoke(prompt_args! {"a" => 1}).await.unwrap(), "7");
    }

    #[tokio::test]
    async fn checks_inputs_before_and_outputs_after() {
        let calls = Arc::new(Mutex::new(0));
        let counter = calls.clone();
        let chain = TransformChain::new(["a", "b"], ["x", "y"], move |_| {
            *counter.lock().unwrap() += 1;
            async { Ok(out(&[("x", json!(1))])) }
        });
        let err = chain.call(prompt_args! {"a" => 1}).await.unwrap_err();
        assert!(
            matches!(&err, ChainError::MissingInputVariable(k) if k == "b"),
            "{err:?}"
        );
        assert_eq!(*calls.lock().unwrap(), 0, "the function must not run");

        let err = chain
            .execute(prompt_args! {"a" => 1, "b" => 2})
            .await
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "Error: transform did not return output key `y`"
        );
        assert_eq!(*calls.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn function_errors_pass_through() {
        let chain = TransformChain::new(Vec::<String>::new(), ["x"], |_| async {
            Err(ChainError::OtherError("bad input".into()))
        });
        let err = chain.invoke(prompt_args! {}).await.unwrap_err();
        assert_eq!(err.to_string(), "Error: bad input");
    }

    #[tokio::test]
    async fn no_output_keys_means_an_empty_generation() {
        let chain = TransformChain::new(Vec::<String>::new(), Vec::<String>::new(), |_| async {
            Ok(out(&[("side", json!(true))]))
        });
        assert_eq!(chain.invoke(prompt_args! {}).await.unwrap(), "");
        let executed = chain.execute(prompt_args! {}).await.unwrap();
        assert_eq!(executed["side"], json!(true));
    }

    #[tokio::test]
    async fn works_inside_a_sequential_chain() {
        let llm = FakeLLM::new(["a short poem"]);
        let write = LLMChainBuilder::new()
            .prompt(HumanMessagePromptTemplate::new(template_fstring!(
                "Write about {upper}",
                "upper"
            )))
            .llm(llm.clone())
            .output_key("poem")
            .build()
            .unwrap();
        let chain = sequential_chain!(upper(), write);
        let executed = chain
            .execute(prompt_args! {"text" => "rust"})
            .await
            .unwrap();
        assert_eq!(executed["upper"], json!("RUST"));
        assert_eq!(executed["poem"], json!("a short poem"));
        assert_eq!(llm.calls()[0][0].content, "Write about RUST");
        // The sequence's result is the last chain's, with the LLM's token usage.
        let result: GenerateResult =
            serde_json::from_value(executed[DEFAULT_RESULT_KEY].clone()).unwrap();
        assert_eq!(result.generation, "a short poem");
        assert!(result.tokens.is_some());
    }

    #[tokio::test]
    async fn streams_through_the_default_call_stream() {
        let chain = upper();
        let events: Vec<_> = chain
            .call_stream(prompt_args! {"text" => "ok"})
            .collect()
            .await;
        let texts: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                Ok(LLMEvent::Text(t)) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["OK"]);
    }
}
