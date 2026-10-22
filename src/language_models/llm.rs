use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::schemas::{Message, StreamData};

use super::{
    box_llm_events, options::CallOptions, result_events, GenerateResult, LLMError, LLMEventStream,
};

#[async_trait]
pub trait LLM: Sync + Send + LLMClone {
    async fn generate(&self, messages: &[Message]) -> Result<GenerateResult, LLMError>;
    async fn invoke(&self, prompt: &str) -> Result<String, LLMError> {
        self.generate(&[Message::new_human_message(prompt)])
            .await
            .map(|res| res.generation)
    }
    async fn stream(
        &self,
        _messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError>;

    /// Streams the response as [`LLMEvent`](super::LLMEvent)s: `Text` chunks while the
    /// model writes, then one `Done` with the complete [`GenerateResult`] (including typed
    /// `tool_calls`). Errors, including the initial request failing, come as stream items.
    ///
    /// The default implementation calls [`generate`](LLM::generate) and emits its result
    /// as a single chunk, so every LLM supports it; providers override it to stream for real.
    ///
    /// ```rust,ignore
    /// let mut events = llm.generate_stream(&messages);
    /// while let Some(event) = events.next().await {
    ///     match event? {
    ///         LLMEvent::Text(chunk) => print!("{chunk}"),
    ///         LLMEvent::Done(result) => println!("\n{:?}", result.tokens),
    ///         _ => {}
    ///     }
    /// }
    /// ```
    fn generate_stream<'a>(&'a self, messages: &'a [Message]) -> LLMEventStream<'a> {
        box_llm_events(async_stream::try_stream! {
            let result = self.generate(messages).await?;
            for event in result_events(result) {
                yield event;
            }
        })
    }

    /// This is usefull when you want to create a chain and override
    /// LLM options
    fn add_options(&mut self, _options: CallOptions) {
        // No action taken
    }
    //This is usefull when using non chat models
    fn messages_to_string(&self, messages: &[Message]) -> String {
        messages
            .iter()
            .map(|m| format!("{:?}: {}", m.message_type, m.content))
            .collect::<Vec<String>>()
            .join("\n")
    }
}

pub trait LLMClone {
    fn clone_box(&self) -> Box<dyn LLM>;
}

impl<T> LLMClone for T
where
    T: 'static + LLM + Clone,
{
    fn clone_box(&self) -> Box<dyn LLM> {
        Box::new(self.clone())
    }
}

impl<L> From<L> for Box<dyn LLM>
where
    L: 'static + LLM,
{
    fn from(llm: L) -> Self {
        Box::new(llm)
    }
}
