use futures::StreamExt;
use serde_json::{json, Value};

use crate::{
    chain::{
        stuff_documents::{
            input_documents, COMBINE_DOCUMENTS_DEFAULT_DOCUMENT_VARIABLE_NAME,
            COMBINE_DOCUMENTS_DEFAULT_INPUT_KEY,
        },
        Chain, ChainError, ChainEventStream,
    },
    language_models::{GenerateResult, LLMEvent, TokenUsage},
    prompt::PromptArgs,
    schemas::Document,
};

/// Adds `usage` to the running total, if the provider reported any.
pub(super) fn add_usage(total: &mut Option<TokenUsage>, usage: &Option<TokenUsage>) {
    if let Some(usage) = usage {
        total.get_or_insert_with(TokenUsage::default).add(usage);
    }
}

/// `input_variables` without the documents key: the extra inputs (e.g. `question`) that
/// every step receives.
pub(super) fn shared_inputs(input_variables: &PromptArgs, input_key: &str) -> PromptArgs {
    let mut rest = input_variables.clone();
    rest.remove(input_key);
    rest
}

/// Runs `chain` with `call_stream`, adding `usage` (from earlier steps) to the final result.
pub(super) fn stream_last_step<'a>(
    chain: &'a dyn Chain,
    inputs: PromptArgs,
    usage: Option<TokenUsage>,
) -> ChainEventStream<'a> {
    Box::pin(chain.call_stream(inputs).map(move |event| match event {
        Ok(LLMEvent::Done(mut result)) => {
            let mut total = usage.clone();
            add_usage(&mut total, &result.tokens);
            result.tokens = total;
            Ok(LLMEvent::Done(result))
        }
        other => other,
    }))
}

/// Combines documents by running a chain on the first one, then refining that answer
/// with each following document in turn.
///
/// Good when the documents don't fit in one prompt and order matters, or when you want
/// the answer to build up as the model reads. It makes one LLM call per document,
/// sequentially.
///
/// Inputs: `input_documents` (a `Vec<Document>`) plus whatever the prompts use (e.g.
/// `question`). The initial chain gets each document as `context`; the refine chain also
/// gets the previous answer as `existing_answer`.
///
/// ```rust,ignore
/// let chain = RefineDocuments::load_summarize(OpenAI::default());
/// let summary = chain
///     .invoke(prompt_args! { "input_documents" => docs })
///     .await?;
/// ```
pub struct RefineDocuments {
    initial_chain: Box<dyn Chain>,
    refine_chain: Box<dyn Chain>,
    input_key: String,
    document_variable_name: String,
    initial_response_name: String,
}

impl RefineDocuments {
    /// `initial_chain` answers from the first document; `refine_chain` takes
    /// `existing_answer` and the next document and returns an improved answer.
    pub fn new<C1: Into<Box<dyn Chain>>, C2: Into<Box<dyn Chain>>>(
        initial_chain: C1,
        refine_chain: C2,
    ) -> Self {
        Self {
            initial_chain: initial_chain.into(),
            refine_chain: refine_chain.into(),
            input_key: COMBINE_DOCUMENTS_DEFAULT_INPUT_KEY.into(),
            document_variable_name: COMBINE_DOCUMENTS_DEFAULT_DOCUMENT_VARIABLE_NAME.into(),
            initial_response_name: "existing_answer".into(),
        }
    }

    /// Input key holding the documents (default `input_documents`).
    pub fn with_input_key<S: Into<String>>(mut self, key: S) -> Self {
        self.input_key = key.into();
        self
    }

    /// Prompt variable each document's text is passed as (default `context`).
    pub fn with_document_variable_name<S: Into<String>>(mut self, name: S) -> Self {
        self.document_variable_name = name.into();
        self
    }

    /// Prompt variable the previous answer is passed as (default `existing_answer`).
    pub fn with_initial_response_name<S: Into<String>>(mut self, name: S) -> Self {
        self.initial_response_name = name.into();
        self
    }

    fn step_inputs(&self, rest: &PromptArgs, doc: &Document, answer: Option<&str>) -> PromptArgs {
        let mut inputs = rest.clone();
        inputs.insert(
            self.document_variable_name.clone(),
            Value::String(doc.page_content.clone()),
        );
        if let Some(answer) = answer {
            inputs.insert(self.initial_response_name.clone(), json!(answer));
        }
        inputs
    }

    /// Runs every step but the last one, returning the last step's chain, its inputs and
    /// the usage so far.
    async fn run_until_last(
        &self,
        input_variables: &PromptArgs,
    ) -> Result<(&dyn Chain, PromptArgs, Option<TokenUsage>), ChainError> {
        let docs = input_documents(input_variables, &self.input_key)?;
        let Some((last, init)) = docs.split_last() else {
            return Err(ChainError::OtherError(format!(
                "{} must contain at least one document",
                self.input_key
            )));
        };
        let rest = shared_inputs(input_variables, &self.input_key);

        let mut usage = None;
        let mut answer: Option<String> = None;
        for doc in init {
            let (chain, inputs) = self.step(&rest, doc, answer.as_deref());
            let result = chain.call(inputs).await?;
            add_usage(&mut usage, &result.tokens);
            answer = Some(result.generation);
        }
        let (chain, inputs) = self.step(&rest, last, answer.as_deref());
        Ok((chain, inputs, usage))
    }

    /// The first document goes to the initial chain, the rest to the refine chain.
    fn step(
        &self,
        rest: &PromptArgs,
        doc: &Document,
        answer: Option<&str>,
    ) -> (&dyn Chain, PromptArgs) {
        let chain = match answer {
            None => self.initial_chain.as_ref(),
            Some(_) => self.refine_chain.as_ref(),
        };
        (chain, self.step_inputs(rest, doc, answer))
    }
}

#[async_trait::async_trait]
impl Chain for RefineDocuments {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        let (chain, inputs, mut usage) = self.run_until_last(&input_variables).await?;
        let mut result = chain.call(inputs).await?;
        add_usage(&mut usage, &result.tokens);
        result.tokens = usage;
        Ok(result)
    }

    /// Runs the intermediate steps quietly and streams the final refinement.
    fn call_stream<'a>(&'a self, input_variables: PromptArgs) -> ChainEventStream<'a> {
        Box::pin(
            futures::stream::once(async move {
                match self.run_until_last(&input_variables).await {
                    Ok((chain, inputs, usage)) => stream_last_step(chain, inputs, usage),
                    Err(e) => Box::pin(futures::stream::iter([Err(e)])) as ChainEventStream<'a>,
                }
            })
            .flatten(),
        )
    }

    fn get_input_keys(&self) -> Vec<String> {
        let mut keys = vec![self.input_key.clone()];
        keys.extend(
            self.initial_chain
                .get_input_keys()
                .into_iter()
                .filter(|k| *k != self.document_variable_name),
        );
        keys
    }
}
