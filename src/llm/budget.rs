//! USD spend caps for any [`LLM`], with budget slices for sub-chains and sub-agents.
//!
//! [`BudgetedLLM`] reserves the most a call can cost before it is sent, refuses the call
//! when that does not fit in what is left, and books what the provider reports when the
//! call returns. Reservations go through one atomic ledger
//! ([`llm_budget::AgentBudget`]), so chains or agents running in parallel on the same
//! budget can never be admitted for more than it holds.
//!
//! [`BudgetedLLM::sub_budget`] carves a slice out of the budget for one sub-chain or
//! sub-agent. The slice is reserved up front, everything spent in it is booked on the
//! parent at once, and the unused part goes back when the last clone of the sub-budget
//! LLM is dropped:
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use langchain_rust::llm::{BudgetedLLM, Limit, OpenAI, OpenAIModel};
//! use llm_budget::AgentBudget;
//!
//! let model = OpenAIModel::Gpt4oMini.to_string();
//! let budget = Arc::new(AgentBudget::builder("pipeline", 0.05).build()?);
//! let llm = BudgetedLLM::new(OpenAI::default().with_model(&model), budget, &model);
//!
//! // Two sub-chains in parallel: each gets its own slice, together at most $0.05.
//! let research = llm.sub_budget("research", Limit::Usd(0.03))?;
//! let summary = llm.sub_budget("summary", Limit::Fraction(1.0))?; // the $0.02 left
//! ```
//!
//! Enable with the `budget` feature. Prices come from llm-budget's built-in table; a model
//! it has no price for is refused unless the budget has an
//! [`UnpricedPolicy`](llm_budget::UnpricedPolicy).

use std::{pin::Pin, sync::Arc};

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use llm_budget::{AgentBudget, Reservation, ScopeGuard, Usage};

pub use llm_budget::Limit;

use crate::{
    language_models::{
        box_llm_events, count_tokens, llm::LLM, options::CallOptions, LLMError, LLMEvent,
        LLMEventStream, TokenUsage,
    },
    schemas::{Message, StreamData},
};

/// Output tokens reserved per call when no `max_tokens` is set.
const DEFAULT_OUTPUT_RESERVE: u32 = 1_024;

/// Wraps an [`LLM`] so every call is reserved against, and booked on, a shared
/// [`AgentBudget`]. See the [module docs](self).
pub struct BudgetedLLM {
    llm: Box<dyn LLM>,
    budget: Arc<AgentBudget>,
    model: String,
    output_reserve: u32,
    max_tokens: Option<u32>,
    /// For a [`sub_budget`](BudgetedLLM::sub_budget): the slice reserved from the parent,
    /// given back when the last clone is dropped.
    scope: Option<Arc<ScopeGuard>>,
}

impl Clone for BudgetedLLM {
    fn clone(&self) -> Self {
        Self {
            llm: self.llm.clone_box(),
            budget: Arc::clone(&self.budget),
            model: self.model.clone(),
            output_reserve: self.output_reserve,
            max_tokens: self.max_tokens,
            scope: self.scope.clone(),
        }
    }
}

impl BudgetedLLM {
    /// Caps `llm` at `budget`, pricing calls as `model` (the id llm-budget's price table
    /// knows it by, e.g. `"gpt-4o-mini"` or `"claude-sonnet-4-5"`).
    pub fn new<L: LLM + 'static>(
        llm: L,
        budget: Arc<AgentBudget>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            llm: Box::new(llm),
            budget,
            model: model.into(),
            output_reserve: DEFAULT_OUTPUT_RESERVE,
            max_tokens: None,
            scope: None,
        }
    }

    /// Output tokens to reserve per call when the call options set no `max_tokens`
    /// (default 1024). The reservation is the prompt estimate plus this; the booking is
    /// what the provider reports.
    pub fn with_output_reserve(mut self, tokens: u32) -> Self {
        self.output_reserve = tokens;
        self
    }

    /// The budget calls are booked on: spend, remaining, audit log and kill switch.
    pub fn budget(&self) -> &Arc<AgentBudget> {
        &self.budget
    }

    /// The same LLM on a slice of this budget, for one sub-chain or sub-agent.
    ///
    /// The slice (`Limit::Usd(x)`, or `Limit::Fraction(f)` of what is left) is reserved
    /// now, so slices handed to parallel branches can never add up to more than this
    /// budget holds. Calls through the slice are booked on this budget as they settle,
    /// and the unused part comes back when the last clone of the returned LLM is
    /// dropped. Slices can have slices.
    pub fn sub_budget(&self, name: impl Into<String>, limit: Limit) -> Result<Self, LLMError> {
        let scope = self.budget.child(name, limit)?;
        Ok(Self {
            llm: self.llm.clone_box(),
            budget: Arc::clone(scope.budget()),
            model: self.model.clone(),
            output_reserve: self.output_reserve,
            max_tokens: self.max_tokens,
            scope: Some(Arc::new(scope)),
        })
    }

    fn reserve(&self, messages: &[Message]) -> Result<Hold, LLMError> {
        let prompt = count_tokens(&self.model, &self.llm.messages_to_string(messages)) as u64;
        let output = u64::from(self.max_tokens.unwrap_or(self.output_reserve));
        let reservation = self
            .budget
            .reserve_for(&self.model, &Usage::new(prompt, output))?;
        Ok(Hold {
            reservation: Some(reservation),
            model: self.model.clone(),
        })
    }
}

/// A reservation for one call. Settled from the reported usage when the call returns,
/// released when it fails before producing anything. Dropped any other way (cancelled,
/// a stream abandoned or failing midway, no usage reported) it books the reserved
/// estimate, because the provider may already be billing for the call.
struct Hold {
    reservation: Option<Reservation>,
    model: String,
}

impl Hold {
    /// Books the reported usage; with no usage reported, the reserved estimate (on drop).
    fn settle(mut self, tokens: Option<&TokenUsage>) {
        let Some(t) = tokens else { return };
        if let Some(reservation) = self.reservation.take() {
            let usage = Usage {
                input_tokens: t.prompt_tokens.into(),
                cached_input_tokens: t.cache_read_tokens.into(),
                cache_write_tokens: t.cache_creation_tokens.into(),
                output_tokens: t.completion_tokens.into(),
            };
            let _ = reservation.settle_usage(&self.model, &usage);
        }
    }

    fn release(mut self) {
        if let Some(reservation) = self.reservation.take() {
            reservation.release();
        }
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        if let Some(reservation) = self.reservation.take() {
            let estimate = reservation.estimate_usd();
            let _ = reservation.settle(estimate);
        }
    }
}

#[async_trait]
impl LLM for BudgetedLLM {
    async fn generate(
        &self,
        messages: &[Message],
    ) -> Result<crate::language_models::GenerateResult, LLMError> {
        let hold = self.reserve(messages)?;
        match self.llm.generate(messages).await {
            Ok(result) => {
                hold.settle(result.tokens.as_ref());
                Ok(result)
            }
            Err(e) => {
                hold.release();
                Err(e)
            }
        }
    }

    async fn stream(
        &self,
        messages: &[Message],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamData, LLMError>> + Send>>, LLMError> {
        let hold = self.reserve(messages)?;
        let mut inner = match self.llm.stream(messages).await {
            Ok(inner) => inner,
            Err(e) => {
                hold.release();
                return Err(e);
            }
        };
        Ok(Box::pin(async_stream::stream! {
            let mut hold = Some(hold);
            let mut started = false;
            let mut usage: Option<TokenUsage> = None;
            while let Some(chunk) = inner.next().await {
                match &chunk {
                    Ok(data) => {
                        started = true;
                        if data.tokens.is_some() {
                            usage = data.tokens.clone();
                        }
                    }
                    Err(_) if !started => {
                        if let Some(hold) = hold.take() {
                            hold.release();
                        }
                    }
                    Err(_) => {}
                }
                yield chunk;
            }
            if let Some(hold) = hold.take() {
                hold.settle(usage.as_ref());
            }
        }))
    }

    fn generate_stream<'a>(&'a self, messages: &'a [Message]) -> LLMEventStream<'a> {
        box_llm_events(async_stream::stream! {
            let hold = match self.reserve(messages) {
                Ok(hold) => hold,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };
            let mut hold = Some(hold);
            let mut started = false;
            let mut events = self.llm.generate_stream(messages);
            while let Some(event) = events.next().await {
                match &event {
                    Ok(LLMEvent::Done(result)) => {
                        if let Some(hold) = hold.take() {
                            hold.settle(result.tokens.as_ref());
                        }
                    }
                    Ok(_) => started = true,
                    Err(_) if !started => {
                        if let Some(hold) = hold.take() {
                            hold.release();
                        }
                    }
                    Err(_) => {}
                }
                yield event;
            }
        })
    }

    fn add_options(&mut self, options: CallOptions) {
        if options.max_tokens.is_some() {
            self.max_tokens = options.max_tokens;
        }
        self.llm.add_options(options);
    }

    fn messages_to_string(&self, messages: &[Message]) -> String {
        self.llm.messages_to_string(messages)
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    use super::*;
    use crate::{
        language_models::{GenerateResult, LLMEvent},
        test_utils::FakeLLM,
    };

    const MODEL: &str = "gpt-4o-mini";

    fn budget(usd: f64) -> Arc<AgentBudget> {
        Arc::new(AgentBudget::builder("test", usd).build().unwrap())
    }

    fn answer(prompt_tokens: u32, completion_tokens: u32) -> GenerateResult {
        GenerateResult {
            generation: "ok".into(),
            tokens: Some(TokenUsage::new(prompt_tokens, completion_tokens)),
            ..Default::default()
        }
    }

    fn cost(prompt_tokens: u64, completion_tokens: u64) -> f64 {
        budget(1.0)
            .cost_of(MODEL, &Usage::new(prompt_tokens, completion_tokens))
            .unwrap()
    }

    #[tokio::test]
    async fn books_the_reported_usage_and_frees_the_hold() {
        let budget = budget(1.0);
        let llm = BudgetedLLM::new(
            FakeLLM::with_results([answer(1_000, 200)]),
            budget.clone(),
            MODEL,
        );
        llm.invoke("hello").await.unwrap();
        assert!((budget.spent_usd() - cost(1_000, 200)).abs() < 1e-12);
        assert_eq!(budget.reserved_usd(), 0.0);
    }

    #[tokio::test]
    async fn refuses_a_call_that_does_not_fit_before_sending_it() {
        let fake = FakeLLM::with_results([answer(1, 1)]);
        // 1024 output tokens of gpt-4o-mini cost more than this.
        let llm = BudgetedLLM::new(fake.clone(), budget(0.000_1), MODEL);
        let err = llm.invoke("hello").await.unwrap_err();
        assert!(matches!(err, LLMError::BudgetError(_)), "{err}");
        assert!(fake.calls().is_empty());
    }

    #[tokio::test]
    async fn max_tokens_sets_the_output_reservation() {
        let fake = FakeLLM::with_results([answer(1, 1)]);
        let mut llm = BudgetedLLM::new(fake.clone(), budget(0.000_1), MODEL);
        llm.add_options(CallOptions::new().with_max_tokens(16));
        llm.invoke("hello").await.unwrap();
        assert_eq!(fake.calls().len(), 1);
    }

    #[tokio::test]
    async fn a_failed_call_books_nothing() {
        let budget = budget(1.0);
        let llm = BudgetedLLM::new(FakeLLM::new(Vec::<String>::new()), budget.clone(), MODEL);
        assert!(llm.invoke("hello").await.is_err());
        assert_eq!(budget.spent_usd(), 0.0);
        assert_eq!(budget.reserved_usd(), 0.0);
    }

    #[tokio::test]
    async fn generate_stream_settles_on_done() {
        let budget = budget(1.0);
        let llm = BudgetedLLM::new(
            FakeLLM::with_results([answer(500, 50)]),
            budget.clone(),
            MODEL,
        );
        let messages = [Message::new_human_message("hello")];
        let events: Vec<_> = llm.generate_stream(&messages).collect().await;
        assert!(matches!(events.last(), Some(Ok(LLMEvent::Done(_)))));
        assert!((budget.spent_usd() - cost(500, 50)).abs() < 1e-12);
        assert_eq!(budget.reserved_usd(), 0.0);
    }

    #[tokio::test]
    async fn a_stream_without_usage_books_the_estimate() {
        let budget = budget(1.0);
        // FakeLLM's `stream` reports no token usage.
        let llm = BudgetedLLM::new(FakeLLM::new(["a few words"]), budget.clone(), MODEL);
        let stream = llm
            .stream(&[Message::new_human_message("hello")])
            .await
            .unwrap();
        let chunks: Vec<_> = stream.collect().await;
        assert_eq!(chunks.len(), 3);
        assert!(budget.spent_usd() > 0.0);
        assert_eq!(budget.reserved_usd(), 0.0);
    }

    #[tokio::test]
    async fn sub_budgets_book_on_the_parent_and_refund_on_drop() {
        let parent = budget(1.0);
        let llm = BudgetedLLM::new(
            FakeLLM::with_results([answer(1_000, 200)]),
            parent.clone(),
            MODEL,
        );
        let branch = llm.sub_budget("branch", Limit::Usd(0.4)).unwrap();
        assert!((parent.remaining_usd() - 0.6).abs() < 1e-9);
        // A second slice that does not fit in the $0.60 left is refused up front.
        assert!(llm.sub_budget("too-big", Limit::Usd(0.7)).is_err());

        branch.invoke("hello").await.unwrap();
        assert!((parent.spent_usd() - cost(1_000, 200)).abs() < 1e-12);
        drop(branch);
        assert!((parent.remaining_usd() - (1.0 - cost(1_000, 200))).abs() < 1e-9);
    }

    #[tokio::test]
    async fn a_sub_budget_caps_its_branch_without_stopping_the_parent() {
        let parent = budget(1.0);
        let fake = FakeLLM::with_results([answer(1, 1), answer(1, 1)]);
        let llm = BudgetedLLM::new(fake.clone(), parent.clone(), MODEL);
        let branch = llm.sub_budget("branch", Limit::Usd(0.000_1)).unwrap();
        assert!(matches!(
            branch.invoke("hello").await,
            Err(LLMError::BudgetError(_))
        ));
        llm.invoke("hello").await.unwrap();
        assert_eq!(fake.calls().len(), 1);
    }

    #[test]
    fn parallel_sub_budgets_never_overcommit_the_parent() {
        // 16 threads ask for a $0.10 slice of $1.00 at the same moment: exactly 10 fit.
        for _ in 0..200 {
            let parent = budget(1.0);
            let llm = BudgetedLLM::new(FakeLLM::default(), parent.clone(), MODEL);
            let start = Arc::new(std::sync::Barrier::new(16));
            let handles: Vec<_> = (0..16)
                .map(|i| {
                    let llm = llm.clone();
                    let start = Arc::clone(&start);
                    std::thread::spawn(move || {
                        start.wait();
                        llm.sub_budget(format!("branch-{i}"), Limit::Usd(0.1)).ok()
                    })
                })
                .collect();
            let admitted: Vec<BudgetedLLM> = handles
                .into_iter()
                .filter_map(|h| h.join().unwrap())
                .collect();
            assert_eq!(admitted.len(), 10);
            assert!(parent.remaining_usd() < 1e-9);
            drop(admitted);
            assert!((parent.remaining_usd() - 1.0).abs() < 1e-9);
        }
    }
}
