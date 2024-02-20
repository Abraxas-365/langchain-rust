use std::error::Error;

use async_trait::async_trait;

use crate::{
    language_models::{options::CallOptions, GenerateResult},
    prompt::PromptArgs,
};

use super::options::ChainCallOptions;

#[async_trait]
pub trait Chain: Sync + Send {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, Box<dyn Error>>;
    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>>;
}
