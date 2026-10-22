//! Constitutional chain: critiques and revises a chain's answer against a list of
//! principles, ported from langchaingo / LangChain.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    chain::{Chain, ChainError, LLMChain, DEFAULT_OUTPUT_KEY, DEFAULT_RESULT_KEY},
    language_models::{llm::LLM, GenerateResult, TokenUsage},
    output_parsers::strip_reasoning,
    prompt::{PromptArgs, PromptTemplate, TemplateFormat},
    prompt_args,
};

use super::LLMChainBuilder;

mod principles;
mod prompts;

/// Output key with the chain's first answer, when
/// [`with_return_intermediate_steps`](ConstitutionalChain::with_return_intermediate_steps) is on.
pub const CONSTITUTIONAL_INITIAL_OUTPUT_KEY: &str = "initial_output";
/// Output key with every [`CritiqueRevision`], when
/// [`with_return_intermediate_steps`](ConstitutionalChain::with_return_intermediate_steps) is on.
pub const CONSTITUTIONAL_STEPS_KEY: &str = "critiques_and_revisions";

/// A rule the answer is checked against: what to look for in a critique, and how to
/// revise the answer when the critique finds something.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstitutionalPrinciple {
    pub name: String,
    pub critique_request: String,
    pub revision_request: String,
}

impl ConstitutionalPrinciple {
    pub fn new<C: Into<String>, R: Into<String>>(critique_request: C, revision_request: R) -> Self {
        Self {
            name: "Constitutional Principle".into(),
            critique_request: critique_request.into(),
            revision_request: revision_request.into(),
        }
    }

    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = name.into();
        self
    }

    /// A built-in principle by name (`"harmful1"`, `"illegal"`, `"thoughtful"`,
    /// `"uo-ethics-1"`, ...; see [`builtin_names`](Self::builtin_names)).
    pub fn get(name: &str) -> Option<Self> {
        principles::PRINCIPLES
            .iter()
            .find(|(n, _, _)| *n == name)
            .map(|(name, critique, revision)| Self::new(*critique, *revision).with_name(*name))
    }

    /// Names of the built-in principles, the same set as langchaingo and LangChain.
    pub fn builtin_names() -> impl Iterator<Item = &'static str> {
        principles::PRINCIPLES.iter().map(|(name, _, _)| *name)
    }
}

/// One principle's step: the critique and, if it asked for changes, the revision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CritiqueRevision {
    pub principle: String,
    pub critique: String,
    /// `None` when the critique found nothing to change, or the reviser answered
    /// "No revisions needed".
    pub revision: Option<String>,
}

/// What [`ConstitutionalChain::critique_and_revise`] returns.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstitutionalOutput {
    /// The wrapped chain's answer, before any revision.
    pub initial_output: String,
    /// The answer after every principle.
    pub output: String,
    pub critiques_and_revisions: Vec<CritiqueRevision>,
    /// Usage of all LLM calls.
    pub tokens: Option<TokenUsage>,
}

/// Runs a chain, then for each principle asks the model to critique the answer and, when
/// the critique finds a problem, to revise it. Each revision is what the next principle
/// critiques.
///
/// ```rust,ignore
/// let chain = ConstitutionalChain::new(
///     qa_chain,
///     llm,
///     vec![ConstitutionalPrinciple::get("illegal").unwrap()],
/// );
/// let answer = chain.invoke(prompt_args! { "question" => "..." }).await?;
/// ```
pub struct ConstitutionalChain {
    chain: LLMChain,
    critique_chain: Box<dyn Chain>,
    revision_chain: Box<dyn Chain>,
    principles: Vec<ConstitutionalPrinciple>,
    output_key: String,
    return_intermediate_steps: bool,
}

const CRITIQUE_VARIABLES: [&str; 3] = ["input_prompt", "output_from_model", "critique_request"];
const REVISION_VARIABLES: [&str; 5] = [
    "input_prompt",
    "output_from_model",
    "critique_request",
    "critique",
    "revision_request",
];

impl ConstitutionalChain {
    /// `chain` answers first; `llm` writes the critiques and revisions with the built-in
    /// few-shot prompts.
    pub fn new<L: Into<Box<dyn LLM>>>(
        chain: LLMChain,
        llm: L,
        principles: Vec<ConstitutionalPrinciple>,
    ) -> Self {
        let llm = llm.into();
        Self {
            chain,
            critique_chain: Box::new(llm_chain(
                llm.clone_box(),
                critique_template(),
                &CRITIQUE_VARIABLES,
            )),
            revision_chain: Box::new(llm_chain(llm, revision_template(), &REVISION_VARIABLES)),
            principles,
            output_key: DEFAULT_OUTPUT_KEY.into(),
            return_intermediate_steps: false,
        }
    }

    /// Replaces the critique step. It gets `input_prompt`, `output_from_model` and
    /// `critique_request`; an output containing "No critique needed" skips the revision.
    pub fn with_critique_chain<C: Into<Box<dyn Chain>>>(mut self, chain: C) -> Self {
        self.critique_chain = chain.into();
        self
    }

    /// Replaces the revision step. It gets the critique step's inputs plus `critique` and
    /// `revision_request`.
    pub fn with_revision_chain<C: Into<Box<dyn Chain>>>(mut self, chain: C) -> Self {
        self.revision_chain = chain.into();
        self
    }

    /// Key of the final answer in [`execute`](Chain::execute)'s output (default `output`).
    pub fn with_output_key<S: Into<String>>(mut self, key: S) -> Self {
        self.output_key = key.into();
        self
    }

    /// Also return the first answer and every [`CritiqueRevision`] from
    /// [`execute`](Chain::execute), under [`CONSTITUTIONAL_INITIAL_OUTPUT_KEY`] and
    /// [`CONSTITUTIONAL_STEPS_KEY`].
    pub fn with_return_intermediate_steps(mut self, enabled: bool) -> Self {
        self.return_intermediate_steps = enabled;
        self
    }

    /// Runs the chain and every principle, returning all the steps.
    pub async fn critique_and_revise(
        &self,
        input_variables: PromptArgs,
    ) -> Result<ConstitutionalOutput, ChainError> {
        let input_prompt = self.chain.prompt_text(input_variables.clone())?;
        let first = self.chain.call(input_variables).await?;
        let mut tokens = first.tokens;
        let initial_output = strip_reasoning(&first.generation).trim().to_string();

        let mut response = initial_output.clone();
        let mut steps = Vec::with_capacity(self.principles.len());
        for principle in &self.principles {
            let critique = self
                .critique_chain
                .call(prompt_args! {
                    "input_prompt" => input_prompt,
                    "output_from_model" => response,
                    "critique_request" => principle.critique_request,
                })
                .await?;
            add_usage(&mut tokens, &critique.tokens);
            let critique = parse_critique(&critique.generation);
            if critique.to_lowercase().contains("no critique needed") {
                steps.push(CritiqueRevision {
                    principle: principle.name.clone(),
                    critique,
                    revision: None,
                });
                continue;
            }

            let revision = self
                .revision_chain
                .call(prompt_args! {
                    "input_prompt" => input_prompt,
                    "output_from_model" => response,
                    "critique_request" => principle.critique_request,
                    "critique" => critique,
                    "revision_request" => principle.revision_request,
                })
                .await?;
            add_usage(&mut tokens, &revision.tokens);
            let revision = parse_revision(&revision.generation);
            // The revision prompt tells the model to say so instead of changing nothing.
            let revision = if revision.to_lowercase().starts_with("no revisions needed") {
                None
            } else {
                response = revision.clone();
                Some(revision)
            };
            steps.push(CritiqueRevision {
                principle: principle.name.clone(),
                critique,
                revision,
            });
        }

        Ok(ConstitutionalOutput {
            initial_output,
            output: response,
            critiques_and_revisions: steps,
            tokens,
        })
    }
}

#[async_trait::async_trait]
impl Chain for ConstitutionalChain {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        let out = self.critique_and_revise(input_variables).await?;
        Ok(GenerateResult {
            generation: out.output,
            tokens: out.tokens,
            ..Default::default()
        })
    }

    async fn execute(
        &self,
        input_variables: PromptArgs,
    ) -> Result<HashMap<String, Value>, ChainError> {
        let out = self.critique_and_revise(input_variables).await?;
        let result = GenerateResult {
            generation: out.output.clone(),
            tokens: out.tokens,
            ..Default::default()
        };
        let mut output = HashMap::new();
        output.insert(self.output_key.clone(), json!(out.output));
        output.insert(DEFAULT_RESULT_KEY.to_string(), json!(result));
        if self.return_intermediate_steps {
            output.insert(
                CONSTITUTIONAL_INITIAL_OUTPUT_KEY.to_string(),
                json!(out.initial_output),
            );
            output.insert(
                CONSTITUTIONAL_STEPS_KEY.to_string(),
                json!(out.critiques_and_revisions),
            );
        }
        Ok(output)
    }

    fn get_input_keys(&self) -> Vec<String> {
        self.chain.get_input_keys()
    }

    fn get_output_keys(&self) -> Vec<String> {
        let mut keys = vec![self.output_key.clone(), DEFAULT_RESULT_KEY.to_string()];
        if self.return_intermediate_steps {
            keys.push(CONSTITUTIONAL_INITIAL_OUTPUT_KEY.to_string());
            keys.push(CONSTITUTIONAL_STEPS_KEY.to_string());
        }
        keys
    }
}

fn add_usage(total: &mut Option<TokenUsage>, usage: &Option<TokenUsage>) {
    if let Some(usage) = usage {
        total.get_or_insert_with(TokenUsage::default).add(usage);
    }
}

/// The critique without reasoning blocks, an echoed `Critique:` label, or a
/// `Revision request:` the model may have gone on to write.
fn parse_critique(raw: &str) -> String {
    let critique = strip_reasoning(raw);
    // ASCII lowercasing keeps byte offsets valid for slicing `critique`.
    let lower = critique.to_ascii_lowercase();
    let end = lower.find("revision request:").unwrap_or(critique.len());
    strip_label(&critique[..end], "critique:").to_string()
}

/// The revision without reasoning blocks or an echoed `Revision:` label.
fn parse_revision(raw: &str) -> String {
    strip_label(strip_reasoning(raw), "revision:").to_string()
}

/// `text` trimmed, without a leading `label` (any case), as models often repeat the
/// prompt's last line.
fn strip_label<'a>(text: &'a str, label: &str) -> &'a str {
    let text = text.trim();
    match text.get(..label.len()) {
        Some(start) if start.eq_ignore_ascii_case(label) => text[label.len()..].trim(),
        _ => text,
    }
}

fn llm_chain(llm: Box<dyn LLM>, template: String, variables: &[&str]) -> LLMChain {
    LLMChainBuilder::new()
        .prompt(PromptTemplate::new(
            template,
            variables.iter().map(|v| v.to_string()).collect(),
            TemplateFormat::Jinja2,
        ))
        .llm(llm)
        .build()
        .expect("prompt and llm are set")
}

const EXAMPLE_SEPARATOR: &str = "\n === \n";

fn critique_example(e: &prompts::Example) -> String {
    format!(
        "Human: {}\n\nModel: {}\n\nCritique Request: {}\n\nCritique: {}",
        e.input_prompt, e.output_from_model, e.critique_request, e.critique
    )
}

pub(super) fn critique_template() -> String {
    let mut parts = vec!["Below is a conversation between a human and an AI model. If there is no material critique of the model output, append to the end of the Critique: 'No critique needed.' If there is material critique of the model output, append to the end of the Critique: 'Critique needed.'".to_string()];
    parts.extend(prompts::EXAMPLES.iter().map(critique_example));
    parts.push(
        "Human: {{input_prompt}}\n\nModel: {{output_from_model}}\n\nCritique Request: {{critique_request}}\n\nCritique:"
            .to_string(),
    );
    parts.join(EXAMPLE_SEPARATOR)
}

pub(super) fn revision_template() -> String {
    let mut parts = vec!["Below is a conversation between a human and an AI model.".to_string()];
    parts.extend(prompts::EXAMPLES.iter().map(|e| {
        format!(
            "{}\n\nRevision Request: {}\n\nRevision: {}",
            critique_example(e),
            e.revision_request,
            e.revision
        )
    }));
    parts.push(
        "Human: {{input_prompt}}\n\nModel: {{output_from_model}}\n\nCritique Request: {{critique_request}}\n\nCritique: {{critique}}\n\nIf the critique does not identify anything worth changing, ignore the Revision Request and do not make any revisions. Instead, return \"No revisions needed\".\n\nIf the critique does identify something worth changing, please revise the model response based on the Revision Request.\n\nRevision Request: {{revision_request}}\n\nRevision:"
            .to_string(),
    );
    parts.join(EXAMPLE_SEPARATOR)
}

#[cfg(test)]
mod tests;
