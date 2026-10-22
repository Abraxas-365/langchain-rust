use std::{fmt, sync::Arc};

use super::{
    example_selector::ExampleSelector, FormatPrompter, PromptArgs, PromptError, PromptFromatter,
    PromptTemplate,
};
use crate::schemas::{messages::Message, prompt::PromptValue};

/// Where a [`FewShotPromptTemplate`] gets its examples.
#[derive(Clone)]
enum Examples {
    Fixed(Vec<PromptArgs>),
    Selector(Arc<dyn ExampleSelector>),
}

/// A prompt that shows the model some examples before the actual input: an optional
/// prefix, each example formatted with `example_prompt`, then an optional suffix
/// (usually holding the input), joined by a separator (`"\n\n"` by default).
///
/// Examples are either a fixed list or picked per input by an [`ExampleSelector`]
/// (e.g. the most similar ones). Only the prefix and suffix are formatted with the
/// input, so braces inside examples are left alone.
///
/// ```rust,ignore
/// let prompt = FewShotPromptTemplate::new(
///     template_fstring!("Word: {word}\nAntonym: {antonym}", "word", "antonym"),
///     vec![
///         prompt_args! {"word" => "happy", "antonym" => "sad"},
///         prompt_args! {"word" => "tall", "antonym" => "short"},
///     ],
/// )
/// .with_prefix(template_fstring!("Give the antonym of every input.",))
/// .with_suffix(template_fstring!("Word: {input}\nAntonym:", "input"));
///
/// let chain = LLMChainBuilder::new().prompt(prompt).llm(llm).build()?;
/// chain.invoke(prompt_args! {"input" => "big"}).await?;
/// ```
///
/// With a selector the examples are chosen asynchronously, so format it with
/// [`FormatPrompter::format_prompt_async`] or [`format`](Self::format) (chains do
/// this). The sync [`FormatPrompter::format_prompt`] only works with fixed examples.
#[derive(Clone)]
pub struct FewShotPromptTemplate {
    examples: Examples,
    example_prompt: PromptTemplate,
    prefix: Option<PromptTemplate>,
    suffix: Option<PromptTemplate>,
    separator: String,
}

impl FewShotPromptTemplate {
    /// A few-shot prompt with a fixed list of examples, each formatted with `example_prompt`.
    pub fn new(example_prompt: PromptTemplate, examples: Vec<PromptArgs>) -> Self {
        Self::build(example_prompt, Examples::Fixed(examples))
    }

    /// A few-shot prompt whose examples are chosen per input by `selector`.
    pub fn with_selector<S: ExampleSelector + 'static>(
        example_prompt: PromptTemplate,
        selector: S,
    ) -> Self {
        Self::with_shared_selector(example_prompt, Arc::new(selector))
    }

    /// Like [`with_selector`](Self::with_selector), keeping a handle to the selector
    /// (e.g. to add examples later).
    pub fn with_shared_selector(
        example_prompt: PromptTemplate,
        selector: Arc<dyn ExampleSelector>,
    ) -> Self {
        Self::build(example_prompt, Examples::Selector(selector))
    }

    fn build(example_prompt: PromptTemplate, examples: Examples) -> Self {
        Self {
            examples,
            example_prompt,
            prefix: None,
            suffix: None,
            separator: "\n\n".into(),
        }
    }

    /// Text before the examples, formatted with the input.
    pub fn with_prefix(mut self, prefix: PromptTemplate) -> Self {
        self.prefix = Some(prefix);
        self
    }

    /// Text after the examples, formatted with the input. Usually where the input goes.
    pub fn with_suffix(mut self, suffix: PromptTemplate) -> Self {
        self.suffix = Some(suffix);
        self
    }

    /// What goes between the prefix, each example and the suffix. Defaults to `"\n\n"`.
    pub fn with_separator<S: Into<String>>(mut self, separator: S) -> Self {
        self.separator = separator.into();
        self
    }

    /// The prompt text for `input_variables`, choosing examples with the selector if
    /// there is one.
    pub async fn format(&self, input_variables: PromptArgs) -> Result<String, PromptError> {
        let examples = match &self.examples {
            Examples::Fixed(examples) => examples.clone(),
            Examples::Selector(selector) => selector.select_examples(&input_variables).await?,
        };
        self.assemble(&examples, input_variables)
    }

    /// Formats the prefix, the examples and the suffix and joins the non-empty ones.
    fn assemble(
        &self,
        examples: &[PromptArgs],
        input_variables: PromptArgs,
    ) -> Result<String, PromptError> {
        let mut pieces = Vec::with_capacity(examples.len() + 2);
        if let Some(prefix) = &self.prefix {
            pieces.push(prefix.format(input_variables.clone())?);
        }
        for (i, example) in examples.iter().enumerate() {
            let text = self.example_prompt.format(example.clone()).map_err(|e| {
                PromptError::OtherError(format!("formatting example {}: {e}", i + 1))
            })?;
            pieces.push(text);
        }
        if let Some(suffix) = &self.suffix {
            pieces.push(suffix.format(input_variables)?);
        }
        pieces.retain(|piece| !piece.is_empty());
        Ok(pieces.join(&self.separator))
    }
}

impl fmt::Debug for FewShotPromptTemplate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let examples = match &self.examples {
            Examples::Fixed(examples) => format!("{} fixed", examples.len()),
            Examples::Selector(_) => "selector".to_string(),
        };
        f.debug_struct("FewShotPromptTemplate")
            .field("examples", &examples)
            .field("example_prompt", &self.example_prompt.template())
            .field("prefix", &self.prefix.as_ref().map(|p| p.template()))
            .field("suffix", &self.suffix.as_ref().map(|p| p.template()))
            .field("separator", &self.separator)
            .finish()
    }
}

#[async_trait::async_trait]
impl FormatPrompter for FewShotPromptTemplate {
    /// Fixed examples only; with a selector use
    /// [`format_prompt_async`](FormatPrompter::format_prompt_async).
    fn format_prompt(&self, input_variables: PromptArgs) -> Result<PromptValue, PromptError> {
        match &self.examples {
            Examples::Fixed(examples) => {
                let text = self.assemble(examples, input_variables)?;
                Ok(PromptValue::from_messages(vec![
                    Message::new_human_message(text),
                ]))
            }
            Examples::Selector(_) => Err(PromptError::OtherError(
                "a FewShotPromptTemplate with an example selector has to be formatted with \
                 format_prompt_async (chains already do)"
                    .into(),
            )),
        }
    }

    async fn format_prompt_async(
        &self,
        input_variables: PromptArgs,
    ) -> Result<PromptValue, PromptError> {
        let text = self.format(input_variables).await?;
        Ok(PromptValue::from_messages(vec![
            Message::new_human_message(text),
        ]))
    }

    /// The variables of the prefix and the suffix (examples bring their own values).
    fn get_input_variables(&self) -> Vec<String> {
        let mut variables: Vec<String> = Vec::new();
        for template in self.prefix.iter().chain(self.suffix.iter()) {
            for variable in template.variables() {
                if !variables.contains(&variable) {
                    variables.push(variable);
                }
            }
        }
        variables
    }
}
