//! LLM math chain: the model translates a word problem into an expression, and the
//! [calculator](crate::tools::evaluate_expression) computes it.

use std::{collections::HashMap, sync::LazyLock};

use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};

use crate::{
    chain::{Chain, ChainError, LLMChainBuilder, DEFAULT_RESULT_KEY},
    language_models::{llm::LLM, GenerateResult, TokenUsage},
    output_parsers::strip_reasoning,
    prompt::{PromptArgs, PromptTemplate, TemplateFormat},
    tools::{evaluate_expression, format_number},
};

/// Default input key of [`LLMMathChain`].
pub const LLM_MATH_DEFAULT_INPUT_KEY: &str = "question";
/// Default output key of [`LLMMathChain`], as in langchaingo.
pub const LLM_MATH_DEFAULT_OUTPUT_KEY: &str = "answer";
/// [`execute`](Chain::execute) output key with the evaluated expression, when the model
/// wrote one.
pub const LLM_MATH_EXPRESSION_KEY: &str = "expression";

const PROMPT: &str = r#"Translate a math problem into a single-line arithmetic expression and put it in a ```text block. A calculator evaluates it: it supports + - * / % // ^ (power), parentheses, the constants pi and e, and functions such as sqrt, cbrt, abs, exp, ln, log, log10, sin, cos, tan, floor, ceil, round, min, max, pow and factorial. Don't compute the result yourself.

Question: ${Question with math problem.}
```text
${single line expression that solves the problem}
```

Question: What is 37593 * 67?
```text
37593 * 67
```

Question: What is the fifth root of 37593?
```text
37593 ^ (1 / 5)
```

Question: {{question}}
"#;

/// A fenced block (any language tag); the closing fence may be missing.
static FENCED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)```[A-Za-z0-9_-]*[ \t]*\r?\n?(.*?)(?:```|\z)").expect("valid regex")
});

/// What [`LLMMathChain::solve`] returns.
#[derive(Clone, Debug, PartialEq)]
pub struct MathAnswer {
    /// The formatted result (`"2518731"`, `"0.333333333333"`), or the model's own answer
    /// when it replied with `Answer:` instead of an expression.
    pub answer: String,
    /// The expression the model wrote, if any.
    pub expression: Option<String>,
    pub tokens: Option<TokenUsage>,
}

/// Answers math questions by asking the model for an expression and evaluating it with the
/// built-in [calculator](crate::tools::evaluate_expression), so the arithmetic is exact
/// instead of guessed. Nothing the model writes is executed as code.
///
/// The model's expression goes in a fenced block (any language tag, so ```` ```python ````
/// works too). An `Answer: ...` reply is returned as is.
///
/// ```rust,ignore
/// let chain = LLMMathChain::new(OpenAI::default());
/// let answer = chain
///     .invoke(prompt_args! { "question" => "What is 43 * 10000 / 7324.3?" })
///     .await?; // "58.7086820584"
/// ```
pub struct LLMMathChain {
    llm_chain: Box<dyn Chain>,
    input_key: String,
    output_key: String,
}

impl LLMMathChain {
    /// Uses the built-in prompt, which takes `question`.
    pub fn new<L: Into<Box<dyn LLM>>>(llm: L) -> Self {
        let chain = LLMChainBuilder::new()
            .prompt(PromptTemplate::new(
                PROMPT.to_string(),
                vec![LLM_MATH_DEFAULT_INPUT_KEY.to_string()],
                TemplateFormat::Jinja2,
            ))
            .llm(llm)
            .build()
            .expect("prompt and llm are set");
        Self::from_chain(chain)
    }

    /// Uses your own chain to write the expression. Its output must hold the expression in
    /// a fenced block, or an `Answer:` line.
    pub fn from_chain<C: Into<Box<dyn Chain>>>(chain: C) -> Self {
        Self {
            llm_chain: chain.into(),
            input_key: LLM_MATH_DEFAULT_INPUT_KEY.into(),
            output_key: LLM_MATH_DEFAULT_OUTPUT_KEY.into(),
        }
    }

    /// Input key with the question (default `question`). The prompt still gets it as
    /// `question`.
    pub fn with_input_key<S: Into<String>>(mut self, key: S) -> Self {
        self.input_key = key.into();
        self
    }

    /// Key of the answer in [`execute`](Chain::execute)'s output (default `answer`).
    pub fn with_output_key<S: Into<String>>(mut self, key: S) -> Self {
        self.output_key = key.into();
        self
    }

    /// Asks the model for an expression and evaluates it.
    pub async fn solve(&self, input_variables: PromptArgs) -> Result<MathAnswer, ChainError> {
        let mut inputs = input_variables;
        if self.input_key != LLM_MATH_DEFAULT_INPUT_KEY {
            let question = inputs
                .remove(&self.input_key)
                .ok_or_else(|| ChainError::MissingInputVariable(self.input_key.clone()))?;
            inputs.insert(LLM_MATH_DEFAULT_INPUT_KEY.to_string(), question);
        }
        let result = self.llm_chain.call(inputs).await?;
        let (answer, expression) = parse_math_output(&result.generation)?;
        Ok(MathAnswer {
            answer,
            expression,
            tokens: result.tokens,
        })
    }
}

/// `(answer, expression)` from the model's output.
pub(crate) fn parse_math_output(output: &str) -> Result<(String, Option<String>), ChainError> {
    let output = strip_reasoning(output).trim();
    if let Some(caps) = FENCED.captures(output) {
        let expression = caps[1].trim();
        if !expression.is_empty() {
            let value = evaluate_expression(expression).map_err(|e| {
                ChainError::OtherError(format!("llm math: evaluating `{expression}`: {e}"))
            })?;
            return Ok((format_number(value), Some(expression.to_string())));
        }
    }
    if let Some(start) = output.rfind("Answer:") {
        let answer = output[start + "Answer:".len()..].trim();
        if !answer.is_empty() {
            return Ok((answer.to_string(), None));
        }
    }
    Err(ChainError::OtherError(format!(
        "llm math: expected an expression in a ``` block or an `Answer:` line, got: {output:?}"
    )))
}

#[async_trait]
impl Chain for LLMMathChain {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        let answer = self.solve(input_variables).await?;
        Ok(GenerateResult {
            generation: answer.answer,
            tokens: answer.tokens,
            ..Default::default()
        })
    }

    async fn execute(
        &self,
        input_variables: PromptArgs,
    ) -> Result<HashMap<String, Value>, ChainError> {
        let answer = self.solve(input_variables).await?;
        let result = GenerateResult {
            generation: answer.answer.clone(),
            tokens: answer.tokens,
            ..Default::default()
        };
        let mut output = HashMap::new();
        output.insert(self.output_key.clone(), json!(answer.answer));
        output.insert(DEFAULT_RESULT_KEY.to_string(), json!(result));
        if let Some(expression) = answer.expression {
            output.insert(LLM_MATH_EXPRESSION_KEY.to_string(), json!(expression));
        }
        Ok(output)
    }

    fn get_input_keys(&self) -> Vec<String> {
        vec![self.input_key.clone()]
    }

    fn get_output_keys(&self) -> Vec<String> {
        vec![
            self.output_key.clone(),
            DEFAULT_RESULT_KEY.to_string(),
            LLM_MATH_EXPRESSION_KEY.to_string(),
        ]
    }
}

#[cfg(test)]
mod tests;
