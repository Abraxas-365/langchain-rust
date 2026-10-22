use std::{collections::HashMap, sync::LazyLock};

use futures::{stream, StreamExt, TryStreamExt};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    chain::{
        stuff_documents::{
            input_documents, COMBINE_DOCUMENTS_DEFAULT_DOCUMENT_VARIABLE_NAME,
            COMBINE_DOCUMENTS_DEFAULT_INPUT_KEY,
        },
        Chain, ChainError, DEFAULT_OUTPUT_KEY, DEFAULT_RESULT_KEY,
    },
    language_models::{GenerateResult, TokenUsage},
    prompt::PromptArgs,
};

use super::refine::{add_usage, shared_inputs};

const DEFAULT_MAX_CONCURRENCY: usize = 5;
/// Output key with the winning answer's score (see [`MapRerankDocuments::execute`]).
pub const MAP_RERANK_SCORE_KEY: &str = "score";
/// Output key with every document's answer, when
/// [`with_return_intermediate_steps`](MapRerankDocuments::with_return_intermediate_steps) is on.
pub const MAP_RERANK_INTERMEDIATE_STEPS_KEY: &str = "intermediate_steps";

/// The last `Score: N` in a model output. Tolerates markdown bold and decimals.
static SCORE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\**\s*score\s*\**\s*:\s*\**\s*(\d+(?:\.\d+)?)").expect("valid regex")
});

/// One document's answer, as parsed from the map chain's output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankedAnswer {
    /// The answer text (everything before `Score:`).
    pub answer: String,
    /// How well the answer covers the question, as the model rated it. `None` when the
    /// output had no score.
    pub score: Option<f64>,
    /// Position of the source document in `input_documents`.
    pub document_index: usize,
}

impl RankedAnswer {
    /// Splits `Helpful Answer: ... \nScore: 80` style output. Reasoning in `<think>` tags
    /// and anything before the last `Helpful Answer:` (models often echo the
    /// `Question:` line) are dropped.
    pub fn parse(output: &str, document_index: usize) -> Self {
        let output = match output.rfind("</think>") {
            Some(end) => &output[end + "</think>".len()..],
            None => output,
        };
        let (answer, score) = match SCORE.captures_iter(output).last() {
            Some(caps) => {
                let start = caps.get(0).expect("whole match").start();
                (&output[..start], caps[1].parse::<f64>().ok())
            }
            None => (output, None),
        };
        let answer = match answer.rfind("Helpful Answer:") {
            Some(start) => &answer[start + "Helpful Answer:".len()..],
            None => answer,
        };
        Self {
            answer: answer.trim().to_string(),
            score,
            document_index,
        }
    }
}

/// Answers from every document separately, asking the model to score each answer, and
/// returns the best one.
///
/// Good for questions whose answer sits in one document (a FAQ entry, a product page),
/// where mixing documents would blur it. It makes one LLM call per document, concurrently.
///
/// Inputs: `input_documents` (a `Vec<Document>`) plus whatever the prompt uses (e.g.
/// `question`). The chain gets each document's text as `context` and must end its output
/// with `Score: <0-100>`, like the prompt of [`load_qa`](Self::load_qa) asks.
///
/// Outputs without a score rank below every scored one; if no output has a score the
/// chain fails. Ties keep document order.
///
/// ```rust,ignore
/// let chain = MapRerankDocuments::load_qa(OpenAI::default());
/// let answer = chain
///     .invoke(prompt_args! { "input_documents" => docs, "question" => "Who wrote it?" })
///     .await?;
/// ```
pub struct MapRerankDocuments {
    map_chain: Box<dyn Chain>,
    input_key: String,
    output_key: String,
    document_variable_name: String,
    max_concurrency: usize,
    return_intermediate_steps: bool,
}

impl MapRerankDocuments {
    /// `map_chain` runs once per document and must answer with a trailing `Score:`.
    pub fn new<M: Into<Box<dyn Chain>>>(map_chain: M) -> Self {
        Self {
            map_chain: map_chain.into(),
            input_key: COMBINE_DOCUMENTS_DEFAULT_INPUT_KEY.into(),
            output_key: DEFAULT_OUTPUT_KEY.into(),
            document_variable_name: COMBINE_DOCUMENTS_DEFAULT_DOCUMENT_VARIABLE_NAME.into(),
            max_concurrency: DEFAULT_MAX_CONCURRENCY,
            return_intermediate_steps: false,
        }
    }

    /// Input key holding the documents (default `input_documents`).
    pub fn with_input_key<S: Into<String>>(mut self, key: S) -> Self {
        self.input_key = key.into();
        self
    }

    /// Key of the answer in [`execute`](Chain::execute)'s output (default `output`).
    pub fn with_output_key<S: Into<String>>(mut self, key: S) -> Self {
        self.output_key = key.into();
        self
    }

    /// Prompt variable each document's text goes in (default `context`).
    pub fn with_document_variable_name<S: Into<String>>(mut self, name: S) -> Self {
        self.document_variable_name = name.into();
        self
    }

    /// How many LLM calls run at the same time (default 5, minimum 1).
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency.max(1);
        self
    }

    /// Also return every document's [`RankedAnswer`] (in document order) from
    /// [`execute`](Chain::execute), under [`MAP_RERANK_INTERMEDIATE_STEPS_KEY`].
    pub fn with_return_intermediate_steps(mut self, enabled: bool) -> Self {
        self.return_intermediate_steps = enabled;
        self
    }

    /// Answers from every document and returns the answers in document order, the index
    /// of the best one, and the usage of all calls.
    pub async fn rank(
        &self,
        input_variables: &PromptArgs,
    ) -> Result<(Vec<RankedAnswer>, usize, Option<TokenUsage>), ChainError> {
        let docs = input_documents(input_variables, &self.input_key)?;
        if docs.is_empty() {
            return Err(ChainError::MissingInputVariable(format!(
                "{} (no documents to rank)",
                self.input_key
            )));
        }
        let rest = shared_inputs(input_variables, &self.input_key);
        let map_chain = self.map_chain.as_ref();
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
        // `buffered` keeps the output in document order.
        let results: Vec<GenerateResult> = stream::iter(map_inputs)
            .map(|inputs| map_chain.call(inputs))
            .buffered(self.max_concurrency)
            .try_collect()
            .await?;

        let mut usage = None;
        let answers: Vec<RankedAnswer> = results
            .iter()
            .enumerate()
            .map(|(i, result)| {
                add_usage(&mut usage, &result.tokens);
                RankedAnswer::parse(&result.generation, i)
            })
            .collect();

        let mut best: Option<(usize, f64)> = None;
        for (i, answer) in answers.iter().enumerate() {
            if let Some(score) = answer.score {
                if best.is_none_or(|(_, top)| score > top) {
                    best = Some((i, score));
                }
            }
        }
        let Some((best, _)) = best else {
            let first = results[0].generation.chars().take(200).collect::<String>();
            return Err(ChainError::OtherError(format!(
                "map rerank: no answer had a `Score:` (first output: {first:?})"
            )));
        };
        Ok((answers, best, usage))
    }
}

#[async_trait::async_trait]
impl Chain for MapRerankDocuments {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        let (mut answers, best, tokens) = self.rank(&input_variables).await?;
        Ok(GenerateResult {
            generation: answers.swap_remove(best).answer,
            tokens,
            ..Default::default()
        })
    }

    /// Returns the best answer under the output key, its score under
    /// [`MAP_RERANK_SCORE_KEY`], and optionally every answer under
    /// [`MAP_RERANK_INTERMEDIATE_STEPS_KEY`].
    async fn execute(
        &self,
        input_variables: PromptArgs,
    ) -> Result<HashMap<String, Value>, ChainError> {
        let (answers, best, tokens) = self.rank(&input_variables).await?;
        let winner = &answers[best];
        let result = GenerateResult {
            generation: winner.answer.clone(),
            tokens,
            ..Default::default()
        };
        let mut output = HashMap::new();
        output.insert(self.output_key.clone(), json!(winner.answer));
        output.insert(MAP_RERANK_SCORE_KEY.to_string(), json!(winner.score));
        output.insert(DEFAULT_RESULT_KEY.to_string(), json!(result));
        if self.return_intermediate_steps {
            output.insert(
                MAP_RERANK_INTERMEDIATE_STEPS_KEY.to_string(),
                json!(answers),
            );
        }
        Ok(output)
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

    fn get_output_keys(&self) -> Vec<String> {
        let mut keys = vec![
            self.output_key.clone(),
            MAP_RERANK_SCORE_KEY.to_string(),
            DEFAULT_RESULT_KEY.to_string(),
        ];
        if self.return_intermediate_steps {
            keys.push(MAP_RERANK_INTERMEDIATE_STEPS_KEY.to_string());
        }
        keys
    }
}
