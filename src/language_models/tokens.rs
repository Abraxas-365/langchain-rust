//! Token counting and model context sizes, for fitting text into a model's window.
//!
//! Counts use tiktoken, so they are exact for OpenAI models and an approximation for
//! others. Context sizes come from tiktoken-rs' table of OpenAI models.

use tiktoken_rs::{bpe_for_model, model::get_context_size, o200k_base_singleton};

/// The number of tokens in `text` for `model` (e.g. `gpt-4o`).
///
/// Uses the model's tiktoken encoding, or `o200k_base` (the encoding of current OpenAI
/// models) when tiktoken doesn't know the model. Special tokens such as `<|endoftext|>`
/// are counted as plain text.
///
/// ```rust
/// use langchain_rust::language_models::count_tokens;
///
/// assert_eq!(count_tokens("gpt-4o", "hello world"), 2);
/// ```
///
/// Like langchaingo's `llms.CountTokens`.
pub fn count_tokens(model: &str, text: &str) -> usize {
    bpe_for_model(model)
        .unwrap_or_else(|_| o200k_base_singleton())
        .encode_ordinary(text)
        .len()
}

/// The context window of `model` in tokens, or `None` when it isn't a known OpenAI model.
///
/// Fine-tuned names (`ft:gpt-4o-mini:org::id`) resolve to their base model.
///
/// ```rust
/// use langchain_rust::language_models::model_context_size;
///
/// assert_eq!(model_context_size("gpt-4o"), Some(128_000));
/// assert_eq!(model_context_size("my-local-model"), None);
/// ```
///
/// Like langchaingo's `llms.GetModelContextSize`, which returns 2048 for unknown models.
pub fn model_context_size(model: &str) -> Option<usize> {
    get_context_size(model)
}

/// Tokens left in `model`'s context window after `text`: 0 when the text already fills
/// it, `None` when the context size is unknown (see [`model_context_size`]).
///
/// ```rust
/// use langchain_rust::language_models::calculate_max_tokens;
///
/// assert_eq!(calculate_max_tokens("gpt-4", "hello world"), Some(8192 - 2));
/// ```
///
/// Like langchaingo's `llms.CalculateMaxTokens`.
pub fn calculate_max_tokens(model: &str, text: &str) -> Option<usize> {
    model_context_size(model).map(|size| size.saturating_sub(count_tokens(model, text)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_tokens_uses_the_models_encoding() {
        // "hello world" is 2 tokens in every OpenAI encoding; this text is not.
        let text = "    indented\tcode();";
        let o200k = o200k_base_singleton().encode_ordinary(text).len();
        let cl100k = tiktoken_rs::cl100k_base_singleton()
            .encode_ordinary(text)
            .len();
        let r50k = tiktoken_rs::r50k_base_singleton()
            .encode_ordinary(text)
            .len();
        assert_ne!(cl100k, r50k);
        assert_eq!(count_tokens("gpt-4o", text), o200k);
        assert_eq!(count_tokens("gpt-4", text), cl100k);
        assert_eq!(count_tokens("davinci", text), r50k);
        assert_eq!(count_tokens("gpt-4o", "hello world"), 2);
        assert_eq!(count_tokens("gpt-4o", ""), 0);
    }

    #[test]
    fn count_tokens_falls_back_to_o200k_for_unknown_models() {
        let text = "Ünïcödé text, with 123 numbers and some_identifiers.";
        assert_eq!(
            count_tokens("llama3.1:8b", text),
            o200k_base_singleton().encode_ordinary(text).len()
        );
        assert_eq!(count_tokens("", text), count_tokens("gpt-4o", text));
    }

    #[test]
    fn special_tokens_count_as_text() {
        let special = count_tokens("gpt-4o", "<|endoftext|>");
        assert!(special > 1, "{special}");
    }

    #[test]
    fn context_sizes() {
        assert_eq!(model_context_size("gpt-4"), Some(8192));
        assert_eq!(model_context_size("gpt-4-32k"), Some(32_768));
        assert_eq!(model_context_size("gpt-4o-mini"), Some(128_000));
        assert_eq!(
            model_context_size("ft:gpt-4o-mini:acme::abc123"),
            Some(128_000)
        );
        assert_eq!(model_context_size("qwen3:14b"), None);
    }

    #[test]
    fn max_tokens_is_what_is_left() {
        let text = "one two three four five";
        assert_eq!(
            calculate_max_tokens("gpt-4", text),
            Some(8192 - count_tokens("gpt-4", text))
        );
        assert_eq!(calculate_max_tokens("gpt-4", ""), Some(8192));
        // Never underflows.
        let huge = "word ".repeat(9000);
        assert!(count_tokens("gpt-4", &huge) > 8192);
        assert_eq!(calculate_max_tokens("gpt-4", &huge), Some(0));
        assert_eq!(calculate_max_tokens("qwen3:14b", text), None);
    }
}
