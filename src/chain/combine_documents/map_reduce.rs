use futures::{stream, StreamExt, TryStreamExt};
use serde_json::{json, Value};

use crate::{
    chain::{
        stuff_documents::{
            input_documents, COMBINE_DOCUMENTS_DEFAULT_DOCUMENT_VARIABLE_NAME,
            COMBINE_DOCUMENTS_DEFAULT_INPUT_KEY,
        },
        Chain, ChainError, ChainEventStream,
    },
    language_models::{GenerateResult, TokenUsage},
    prompt::PromptArgs,
    schemas::Document,
};

use super::refine::{add_usage, shared_inputs, stream_last_step};

const DEFAULT_MAX_CONCURRENCY: usize = 5;

/// Combines documents in two phases: a map chain runs on every document independently
/// (concurrently), then a reduce chain combines the mapped outputs.
///
/// Good for documents too large to fit in one prompt together, when each one can be
/// processed on its own (summaries, extracting relevant passages).
///
/// Inputs: `input_documents` (a `Vec<Document>`) plus whatever the prompts use (e.g.
/// `question`). The map chain gets each document's text as `context`. The reduce chain
/// gets the mapped outputs as `input_documents` (keeping each source document's metadata),
/// so a [`StuffDocument`](crate::chain::StuffDocument) works as the reduce chain.
///
/// ```rust,ignore
/// let chain = MapReduceDocuments::load_summarize(OpenAI::default());
/// let summary = chain
///     .invoke(prompt_args! { "input_documents" => docs })
///     .await?;
/// ```
pub struct MapReduceDocuments {
    map_chain: Box<dyn Chain>,
    reduce_chain: Box<dyn Chain>,
    input_key: String,
    document_variable_name: String,
    reduce_input_key: String,
    max_concurrency: usize,
}

impl MapReduceDocuments {
    /// `map_chain` runs once per document; `reduce_chain` combines the results.
    pub fn new<M: Into<Box<dyn Chain>>, R: Into<Box<dyn Chain>>>(
        map_chain: M,
        reduce_chain: R,
    ) -> Self {
        Self {
            map_chain: map_chain.into(),
            reduce_chain: reduce_chain.into(),
            input_key: COMBINE_DOCUMENTS_DEFAULT_INPUT_KEY.into(),
            document_variable_name: COMBINE_DOCUMENTS_DEFAULT_DOCUMENT_VARIABLE_NAME.into(),
            reduce_input_key: COMBINE_DOCUMENTS_DEFAULT_INPUT_KEY.into(),
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
        }
    }

    /// Input key holding the documents (default `input_documents`).
    pub fn with_input_key<S: Into<String>>(mut self, key: S) -> Self {
        self.input_key = key.into();
        self
    }

    /// Prompt variable the map chain receives each document's text as (default `context`).
    pub fn with_document_variable_name<S: Into<String>>(mut self, name: S) -> Self {
        self.document_variable_name = name.into();
        self
    }

    /// Input key the reduce chain receives the mapped documents in (default
    /// `input_documents`).
    pub fn with_reduce_input_key<S: Into<String>>(mut self, key: S) -> Self {
        self.reduce_input_key = key.into();
        self
    }

    /// How many map calls run at the same time (default 5, minimum 1).
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency.max(1);
        self
    }

    /// Runs the map phase and returns the reduce chain's inputs plus the usage so far.
    async fn map(
        &self,
        input_variables: &PromptArgs,
    ) -> Result<(PromptArgs, Option<TokenUsage>), ChainError> {
        let docs = input_documents(input_variables, &self.input_key)?;
        let rest = shared_inputs(input_variables, &self.input_key);

        // `buffered` keeps the output in document order.
        let map_inputs: Vec<PromptArgs> = docs
            .iter()
            .map(|doc| {
                let mut inputs = rest.clone();
                inputs.insert(
                    self.document_variable_name.clone(),
                    Value::String(doc.page_content.clone()),
                );
                inputs
            })
            .collect();
        let map_chain = self.map_chain.as_ref();
        let results: Vec<GenerateResult> = stream::iter(map_inputs)
            .map(|inputs| map_chain.call(inputs))
            .buffered(self.max_concurrency)
            .try_collect()
            .await?;

        let mut usage = None;
        let mapped: Vec<Document> = docs
            .into_iter()
            .zip(results)
            .map(|(doc, result)| {
                add_usage(&mut usage, &result.tokens);
                Document {
                    page_content: result.generation,
                    ..doc
                }
            })
            .collect();

        let mut reduce_inputs = rest;
        reduce_inputs.insert(self.reduce_input_key.clone(), json!(mapped));
        Ok((reduce_inputs, usage))
    }
}

#[async_trait::async_trait]
impl Chain for MapReduceDocuments {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        let (reduce_inputs, mut usage) = self.map(&input_variables).await?;
        let mut result = self.reduce_chain.call(reduce_inputs).await?;
        add_usage(&mut usage, &result.tokens);
        result.tokens = usage;
        Ok(result)
    }

    /// Runs the map phase quietly and streams the reduce step.
    fn call_stream<'a>(&'a self, input_variables: PromptArgs) -> ChainEventStream<'a> {
        Box::pin(
            stream::once(async move {
                match self.map(&input_variables).await {
                    Ok((inputs, usage)) => {
                        stream_last_step(self.reduce_chain.as_ref(), inputs, usage)
                    }
                    Err(e) => Box::pin(stream::iter([Err(e)])) as ChainEventStream<'a>,
                }
            })
            .flatten(),
        )
    }

    fn get_input_keys(&self) -> Vec<String> {
        let mut keys = vec![self.input_key.clone()];
        keys.extend(
            self.map_chain
                .get_input_keys()
                .into_iter()
                .filter(|k| *k != self.document_variable_name),
        );
        keys
    }
}
