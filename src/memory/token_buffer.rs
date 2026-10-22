use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use tiktoken_rs::{bpe_for_model, cl100k_base_singleton};
use tokio::sync::Mutex;

use crate::schemas::{
    memory::{BaseMemory, MemoryError},
    messages::{Message, MessageType},
};

type TokenCounter = Arc<dyn Fn(&str) -> usize + Send + Sync>;

/// Conversation memory that keeps the most recent messages that fit in `max_tokens`.
///
/// When a new message pushes the buffer over the limit, the oldest messages are dropped
/// until it fits again. A tool result left at the front without the AI message that
/// requested it is dropped as well, so the history stays valid for chat APIs.
/// A single message bigger than `max_tokens` is not kept.
///
/// Tokens are counted with tiktoken (`cl100k_base` by default) over each message's
/// content plus its tool calls, which is an approximation for non-OpenAI models.
///
/// ```rust,ignore
/// let memory = TokenBufferMemory::new(2000).with_model("gpt-4o")?;
/// ```
pub struct TokenBufferMemory {
    max_tokens: usize,
    counter: TokenCounter,
    messages: VecDeque<(Message, usize)>,
    total_tokens: usize,
}

impl TokenBufferMemory {
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            counter: Arc::new(|text: &str| cl100k_base_singleton().encode_ordinary(text).len()),
            messages: VecDeque::new(),
            total_tokens: 0,
        }
    }

    /// Count tokens with the tokenizer tiktoken uses for `model` (e.g. `gpt-4o`).
    pub fn with_model(mut self, model: &str) -> Result<Self, MemoryError> {
        let bpe = bpe_for_model(model).map_err(|e| MemoryError::Other(e.to_string()))?;
        self.counter = Arc::new(move |text: &str| bpe.encode_ordinary(text).len());
        Ok(self)
    }

    /// Use a custom token counter, e.g. one matching a non-OpenAI model's tokenizer.
    pub fn with_token_counter<F>(mut self, counter: F) -> Self
    where
        F: Fn(&str) -> usize + Send + Sync + 'static,
    {
        self.counter = Arc::new(counter);
        self
    }

    /// Tokens currently held in the buffer.
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    fn count(&self, message: &Message) -> usize {
        let mut tokens = (self.counter)(&message.content);
        if let Some(tool_calls) = &message.tool_calls {
            tokens += (self.counter)(&tool_calls.to_string());
        }
        tokens
    }

    fn pop_front(&mut self) {
        if let Some((_, tokens)) = self.messages.pop_front() {
            self.total_tokens -= tokens;
        }
    }

    fn prune(&mut self) {
        while self.total_tokens > self.max_tokens && !self.messages.is_empty() {
            self.pop_front();
        }
        while matches!(
            self.messages.front(),
            Some((m, _)) if m.message_type == MessageType::ToolMessage
        ) {
            self.pop_front();
        }
    }
}

impl From<TokenBufferMemory> for Arc<dyn BaseMemory> {
    fn from(memory: TokenBufferMemory) -> Self {
        Arc::new(memory)
    }
}

impl From<TokenBufferMemory> for Arc<Mutex<dyn BaseMemory>> {
    fn from(memory: TokenBufferMemory) -> Self {
        Arc::new(Mutex::new(memory))
    }
}

#[async_trait]
impl BaseMemory for TokenBufferMemory {
    async fn messages(&self) -> Result<Vec<Message>, MemoryError> {
        Ok(self.messages.iter().map(|(m, _)| m.clone()).collect())
    }

    async fn add_message(&mut self, message: Message) -> Result<(), MemoryError> {
        let tokens = self.count(&message);
        self.total_tokens += tokens;
        self.messages.push_back((message, tokens));
        self.prune();
        Ok(())
    }

    async fn clear(&mut self) -> Result<(), MemoryError> {
        self.messages.clear();
        self.total_tokens = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One token per whitespace-separated word, so limits are easy to reason about.
    fn words(max_tokens: usize) -> TokenBufferMemory {
        TokenBufferMemory::new(max_tokens).with_token_counter(|t| t.split_whitespace().count())
    }

    async fn contents(memory: &TokenBufferMemory) -> Vec<String> {
        memory
            .messages()
            .await
            .unwrap()
            .into_iter()
            .map(|m| m.content)
            .collect()
    }

    #[tokio::test]
    async fn keeps_everything_under_the_limit() {
        let mut memory = words(10);
        memory.add_user_message(&"one two three").await.unwrap();
        memory.add_ai_message(&"four five").await.unwrap();
        assert_eq!(memory.total_tokens(), 5);
        assert_eq!(contents(&memory).await, vec!["one two three", "four five"]);
    }

    #[tokio::test]
    async fn drops_oldest_messages_when_over_the_limit() {
        let mut memory = words(5);
        memory.add_user_message(&"a b").await.unwrap();
        memory.add_ai_message(&"c d").await.unwrap();
        memory.add_user_message(&"e f").await.unwrap(); // 6 > 5 -> drop "a b"
        assert_eq!(contents(&memory).await, vec!["c d", "e f"]);
        assert_eq!(memory.total_tokens(), 4);

        memory.add_ai_message(&"g h i j").await.unwrap(); // 8 -> drop "c d", "e f"
        assert_eq!(contents(&memory).await, vec!["g h i j"]);
        assert_eq!(memory.total_tokens(), 4);
    }

    #[tokio::test]
    async fn message_bigger_than_limit_is_not_kept() {
        let mut memory = words(3);
        memory.add_user_message(&"short").await.unwrap();
        memory
            .add_ai_message(&"this one is way too long")
            .await
            .unwrap();
        assert!(contents(&memory).await.is_empty());
        assert_eq!(memory.total_tokens(), 0);
    }

    #[tokio::test]
    async fn tool_calls_count_and_orphan_tool_results_are_dropped() {
        let mut memory = words(6);
        memory
            .add_message(
                Message::new_ai_message("").with_tool_calls(json!([{"id": "call_1", "a": 1}])),
            )
            .await
            .unwrap();
        assert!(memory.total_tokens() > 0, "tool calls must be counted");
        memory
            .add_message(Message::new_tool_message("r1 r2", "call_1"))
            .await
            .unwrap();
        memory.add_ai_message(&"x y z w").await.unwrap();

        // The AI tool-call message is evicted; the tool result must not lead the window.
        let messages = memory.messages().await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].message_type, MessageType::AIMessage);
        assert_eq!(messages[0].content, "x y z w");
        assert_eq!(memory.total_tokens(), 4);
    }

    #[tokio::test]
    async fn clear_resets_tokens() {
        let mut memory = words(10);
        memory.add_user_message(&"a b c").await.unwrap();
        memory.clear().await.unwrap();
        assert_eq!(memory.total_tokens(), 0);
        assert!(contents(&memory).await.is_empty());
    }

    #[tokio::test]
    async fn default_and_model_tokenizers_count_real_tokens() {
        let mut memory = TokenBufferMemory::new(1000);
        memory.add_user_message(&"hello world").await.unwrap();
        assert_eq!(memory.total_tokens(), 2);

        let mut memory = TokenBufferMemory::new(1000).with_model("gpt-4o").unwrap();
        memory.add_user_message(&"hello world").await.unwrap();
        assert_eq!(memory.total_tokens(), 2);

        assert!(TokenBufferMemory::new(10)
            .with_model("not-a-model")
            .is_err());
    }

    #[tokio::test]
    async fn works_behind_the_shared_mutex() {
        let memory: Arc<Mutex<dyn BaseMemory>> = words(2).into();
        memory.lock().await.add_user_message(&"a b").await.unwrap();
        memory.lock().await.add_ai_message(&"c").await.unwrap();
        assert_eq!(memory.lock().await.to_string().await.unwrap(), "ai: c");
    }
}
