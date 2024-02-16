use std::error::Error;

use async_trait::async_trait;

use crate::{language_models::GenerateResult, prompt::PromptArgs};

#[async_trait]
pub trait Chain: Sync + Send {
    async fn call<'a>(
        &'a self,
        input_variables: PromptArgs<'a>,
    ) -> Result<GenerateResult, Box<dyn Error + 'a>>;
    async fn invoke(&self, prompt: &str) -> Result<String, Box<dyn Error>>;
}
