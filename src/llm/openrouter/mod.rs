
//! OpenRouter LLM integration module.
//! 
//! This module provides the OpenRouter client, error types, and model definitions.

pub mod client;
pub mod error;
pub mod models;

pub use client::OpenRouter;
pub use error::OpenRouterError;
pub use models::OpenRouterModel;
