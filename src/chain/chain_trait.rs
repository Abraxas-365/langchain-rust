use std::error::Error;

use async_trait::async_trait;

use crate::{language_models::GenerateResult, prompt::PromptArgs};

#[async_trait]
pub trait Chain: Sync + Send {
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, Box<dyn Error>>;
    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, Box<dyn Error>>;
    fn get_input_keys(&self) -> Vec<String> {
        log::warn!("get_input_keys not implemented for this chain");
        return vec![];
    }
}
