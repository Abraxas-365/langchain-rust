//! Running CPU-bound work (local embeddings, PDF parsing, text splitting) without stalling
//! the async runtime.

/// Runs `f` on Tokio's blocking thread pool so the calling task's worker thread stays free
/// for other tasks, and returns its result. Outside a Tokio runtime (e.g. under
/// `futures::executor::block_on`) it just runs `f` inline.
///
/// A panic in `f` resumes on the caller, as if `f` had run inline.
pub(crate) async fn run_blocking<F, T>(f: F) -> T
where
    F: FnOnce() -> T + Send + 'static,
    T: Send + 'static,
{
    if tokio::runtime::Handle::try_current().is_err() {
        return f();
    }
    match tokio::task::spawn_blocking(f).await {
        Ok(value) => value,
        Err(e) => match e.try_into_panic() {
            Ok(panic) => std::panic::resume_unwind(panic),
            // Only happens when the runtime shuts down under us.
            Err(e) => panic!("blocking task didn't finish: {e}"),
        },
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use std::{
        future::Future,
        sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc,
        },
    };

    /// Runs `work` next to a task that counts how often it gets polled until `work` is done,
    /// on the current (single-threaded) runtime. Returns `work`'s output and that count.
    ///
    /// Work that blocks the thread finishes in its first poll, so the counter never runs
    /// (0); work moved to the blocking pool leaves the thread free and the count is > 0.
    pub(crate) async fn ticks_while<T>(work: impl Future<Output = T>) -> (T, usize) {
        let done = Arc::new(AtomicBool::new(false));
        let ticks = Arc::new(AtomicUsize::new(0));
        let ticker = {
            let (done, ticks) = (done.clone(), ticks.clone());
            async move {
                while !done.load(Ordering::SeqCst) {
                    ticks.fetch_add(1, Ordering::SeqCst);
                    tokio::task::yield_now().await;
                }
            }
        };
        let work = async {
            let out = work.await;
            done.store(true, Ordering::SeqCst);
            out
        };
        let (out, ()) = tokio::join!(work, ticker);
        // The ticker's first poll happens after `work`'s first poll; if `work` was done by
        // then, the ticker exits without counting.
        (out, ticks.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
mod tests {
    use super::{run_blocking, testing::ticks_while};

    #[tokio::test]
    async fn runs_on_another_thread_and_frees_the_runtime() {
        let caller = std::thread::current().id();
        let (worker, ticks) = ticks_while(run_blocking(|| {
            std::thread::sleep(std::time::Duration::from_millis(50));
            std::thread::current().id()
        }))
        .await;
        assert_ne!(worker, caller);
        assert!(ticks > 0, "the runtime thread was blocked");
    }

    #[tokio::test]
    async fn blocking_inline_work_is_detected() {
        let ((), ticks) = ticks_while(async {
            std::thread::sleep(std::time::Duration::from_millis(20));
        })
        .await;
        assert_eq!(ticks, 0);
    }

    #[test]
    fn runs_inline_without_a_runtime() {
        let caller = std::thread::current().id();
        let worker = futures::executor::block_on(run_blocking(|| std::thread::current().id()));
        assert_eq!(worker, caller);
    }

    #[tokio::test]
    async fn panics_resume_on_the_caller() {
        let result = tokio::spawn(run_blocking(|| -> u8 { panic!("boom") })).await;
        let panic = result.unwrap_err().into_panic();
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"boom"));
    }
}
