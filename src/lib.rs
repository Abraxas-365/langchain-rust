#![allow(dead_code)]
//! # Observability
//!
//! The crate is instrumented with [`tracing`]: every LLM call runs in an `llm.generate`
//! span (provider, model, token usage, tool call count, error), embeddings in
//! `embed.documents` / `embed.query`, `LLMChain` in `chain.call`, and agent runs in
//! `agent.run` with an `agent.plan` span per step and a `tool.call` span per tool.
//! Install any subscriber (`tracing-subscriber`, OpenTelemetry, Langfuse...) to see them.
//! Without one, log events are forwarded to the [`log`](https://docs.rs/log) crate, so
//! `env_logger` keeps working.
pub mod agent;
mod blocking;
pub mod chain;
pub mod document_loaders;
pub mod embedding;
pub mod language_models;
pub mod llm;
pub mod memory;
mod observability;
pub mod output_parsers;
pub mod prompt;
pub mod schemas;
pub mod semantic_router;
pub mod text_splitter;
pub mod tools;
pub mod vectorstore;

#[cfg(test)]
pub(crate) mod test_utils;

pub use url;
