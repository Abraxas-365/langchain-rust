use async_trait::async_trait;

use super::OutputParserError;

#[async_trait]
pub trait OutputParser: Send + Sync {
    async fn parse(&self, output: &str) -> Result<String, OutputParserError>;
}

impl<P> From<P> for Box<dyn OutputParser>
where
    P: OutputParser + 'static,
{
    fn from(parser: P) -> Self {
        Box::new(parser)
    }
}
