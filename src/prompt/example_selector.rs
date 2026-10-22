//! Example selectors pick which examples a [`FewShotPromptTemplate`](super::FewShotPromptTemplate)
//! shows for a given input.
use std::{
    cmp::Ordering,
    fmt,
    sync::{Arc, RwLock},
};

use async_trait::async_trait;

use super::{PromptArgs, PromptError, PromptFromatter, PromptTemplate};
use crate::{embedding::Embedder, schemas::messages::value_to_text};

/// Chooses the examples to put in a few-shot prompt for a given input.
///
/// Examples are [`PromptArgs`] maps, formatted with the prompt's example template.
#[async_trait]
pub trait ExampleSelector: Send + Sync {
    /// Adds an example to the pool the selector chooses from.
    async fn add_example(&self, example: PromptArgs) -> Result<(), PromptError>;

    /// The examples to use for `input_variables` (the prompt's inputs), in the order
    /// they should appear.
    async fn select_examples(
        &self,
        input_variables: &PromptArgs,
    ) -> Result<Vec<PromptArgs>, PromptError>;
}

type LengthFn = Arc<dyn Fn(&str) -> usize + Send + Sync>;

/// Selects examples in order until they would no longer fit in `max_length`, counting
/// the length of the input too. Useful to keep a prompt within the context window.
///
/// Length is measured in words by default (see [`with_length_fn`](Self::with_length_fn)).
///
/// ```rust,ignore
/// let selector = LengthBasedExampleSelector::new(
///     template_fstring!("Input: {input}\nOutput: {output}", "input", "output"),
///     vec![prompt_args! {"input" => "happy", "output" => "sad"}],
/// )?
/// .with_max_length(25);
/// ```
pub struct LengthBasedExampleSelector {
    example_prompt: PromptTemplate,
    max_length: usize,
    length_fn: LengthFn,
    /// Each example with the length of its formatted text.
    examples: RwLock<Vec<(PromptArgs, usize)>>,
}

impl LengthBasedExampleSelector {
    /// Default `max_length`, in words.
    pub const DEFAULT_MAX_LENGTH: usize = 2048;

    /// A selector over `examples`, formatted with `example_prompt` to measure them.
    /// Fails if an example is missing a variable of `example_prompt`.
    pub fn new(
        example_prompt: PromptTemplate,
        examples: Vec<PromptArgs>,
    ) -> Result<Self, PromptError> {
        let selector = Self {
            example_prompt,
            max_length: Self::DEFAULT_MAX_LENGTH,
            length_fn: Arc::new(word_count),
            examples: RwLock::new(Vec::new()),
        };
        for example in examples {
            selector.push(example)?;
        }
        Ok(selector)
    }

    /// The most the selected examples plus the input may add up to.
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// How to measure text, e.g. a tokenizer's token count. Example lengths are
    /// recomputed with it.
    pub fn with_length_fn<F>(mut self, length_fn: F) -> Self
    where
        F: Fn(&str) -> usize + Send + Sync + 'static,
    {
        self.length_fn = Arc::new(length_fn);
        let examples = std::mem::take(self.examples.get_mut().unwrap_or_else(|e| e.into_inner()));
        for (example, _) in examples {
            // Every example already formatted once, so this can't fail.
            let _ = self.push(example);
        }
        self
    }

    fn push(&self, example: PromptArgs) -> Result<(), PromptError> {
        let length = (self.length_fn)(&self.example_prompt.format(example.clone())?);
        self.examples
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .push((example, length));
        Ok(())
    }
}

impl fmt::Debug for LengthBasedExampleSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LengthBasedExampleSelector")
            .field("max_length", &self.max_length)
            .field(
                "examples",
                &self
                    .examples
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .len(),
            )
            .finish()
    }
}

#[async_trait]
impl ExampleSelector for LengthBasedExampleSelector {
    async fn add_example(&self, example: PromptArgs) -> Result<(), PromptError> {
        self.push(example)
    }

    async fn select_examples(
        &self,
        input_variables: &PromptArgs,
    ) -> Result<Vec<PromptArgs>, PromptError> {
        let input_length: usize = input_variables
            .values()
            .map(|v| (self.length_fn)(&value_to_text(v)))
            .sum();
        let mut remaining = self.max_length.saturating_sub(input_length);
        let mut selected = Vec::new();
        for (example, length) in self
            .examples
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
        {
            if *length > remaining {
                break;
            }
            remaining -= length;
            selected.push(example.clone());
        }
        Ok(selected)
    }
}

/// Words in `text`, the default length for [`LengthBasedExampleSelector`].
fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

/// Selects the `k` examples most similar to the input, by embedding cosine similarity,
/// most similar first.
///
/// An example's text is its values joined by spaces in key order. With
/// [`with_input_keys`](Self::with_input_keys) only those keys are used, both for the
/// examples and for the input (so the expected output doesn't count).
///
/// ```rust,ignore
/// let selector = SemanticSimilarityExampleSelector::from_examples(
///     examples,
///     OllamaEmbedder::default(),
///     2,
/// )
/// .await?
/// .with_input_keys(["input"]);
/// ```
pub struct SemanticSimilarityExampleSelector {
    embedder: Arc<dyn Embedder>,
    k: usize,
    input_keys: Option<Vec<String>>,
    examples: RwLock<Vec<(PromptArgs, Vec<f64>)>>,
}

impl SemanticSimilarityExampleSelector {
    /// Default number of examples to select.
    pub const DEFAULT_K: usize = 4;

    /// An empty selector; add examples with [`ExampleSelector::add_example`].
    pub fn new<E: Embedder + 'static>(embedder: E) -> Self {
        Self::with_embedder(Arc::new(embedder))
    }

    /// Like [`new`](Self::new), with an embedder shared with other components.
    pub fn with_embedder(embedder: Arc<dyn Embedder>) -> Self {
        Self {
            embedder,
            k: Self::DEFAULT_K,
            input_keys: None,
            examples: RwLock::new(Vec::new()),
        }
    }

    /// A selector over `examples` (embedded in one batch) that picks `k` of them.
    pub async fn from_examples<E: Embedder + 'static>(
        examples: Vec<PromptArgs>,
        embedder: E,
        k: usize,
    ) -> Result<Self, PromptError> {
        let selector = Self::new(embedder).with_k(k);
        selector.add_examples(examples).await?;
        Ok(selector)
    }

    /// How many examples to select.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Only these keys are embedded, for examples and input alike. Examples are
    /// re-embedded when added, so set this before adding them.
    pub fn with_input_keys<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.input_keys = Some(keys.into_iter().map(Into::into).collect());
        self
    }

    /// Adds several examples, embedded in one batch.
    pub async fn add_examples(&self, examples: Vec<PromptArgs>) -> Result<(), PromptError> {
        if examples.is_empty() {
            return Ok(());
        }
        let texts: Vec<String> = examples.iter().map(|e| self.text_of(e)).collect();
        let vectors = self
            .embedder
            .embed_documents(&texts)
            .await
            .map_err(|e| PromptError::OtherError(format!("embedding examples: {e}")))?;
        if vectors.len() != examples.len() {
            return Err(PromptError::OtherError(format!(
                "embedder returned {} vectors for {} examples",
                vectors.len(),
                examples.len()
            )));
        }
        self.examples
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .extend(examples.into_iter().zip(vectors));
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn examples_for_test(&self, examples: Vec<(PromptArgs, Vec<f64>)>) {
        self.examples
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .extend(examples);
    }

    /// The text embedded for `args`: the values of the input keys (or all keys),
    /// in key order, joined by spaces.
    fn text_of(&self, args: &PromptArgs) -> String {
        let mut keys: Vec<&String> = match &self.input_keys {
            Some(keys) => keys.iter().filter(|k| args.contains_key(*k)).collect(),
            None => args.keys().collect(),
        };
        keys.sort();
        keys.iter()
            .map(|k| value_to_text(&args[*k]))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl fmt::Debug for SemanticSimilarityExampleSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SemanticSimilarityExampleSelector")
            .field("k", &self.k)
            .field("input_keys", &self.input_keys)
            .field(
                "examples",
                &self
                    .examples
                    .read()
                    .unwrap_or_else(|e| e.into_inner())
                    .len(),
            )
            .finish()
    }
}

#[async_trait]
impl ExampleSelector for SemanticSimilarityExampleSelector {
    async fn add_example(&self, example: PromptArgs) -> Result<(), PromptError> {
        self.add_examples(vec![example]).await
    }

    async fn select_examples(
        &self,
        input_variables: &PromptArgs,
    ) -> Result<Vec<PromptArgs>, PromptError> {
        if self.k == 0
            || self
                .examples
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
        {
            return Ok(Vec::new());
        }
        let query = self
            .embedder
            .embed_query(&self.text_of(input_variables))
            .await
            .map_err(|e| PromptError::OtherError(format!("embedding the input: {e}")))?;
        let examples = self.examples.read().unwrap_or_else(|e| e.into_inner());
        let mut scored: Vec<(f64, &PromptArgs)> = examples
            .iter()
            .map(|(example, vector)| (cosine_similarity(&query, vector), example))
            .collect();
        // Stable sort: ties keep insertion order.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
        Ok(scored
            .into_iter()
            .take(self.k)
            .map(|(_, example)| example.clone())
            .collect())
    }
}

/// Cosine similarity; 0 when either vector is all zeros.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denominator = norm(a) * norm(b);
    if denominator == 0.0 {
        0.0
    } else {
        dot / denominator
    }
}
