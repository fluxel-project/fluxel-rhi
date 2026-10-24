//! Completion-handler futures for Metal cold object creation.
//!
//! Native callbacks retain the descriptors and functions needed by the call.
//! Dropping a future cancels observation, never a native operation already in
//! progress. The callback drops an unobserved result after completing safely.
use fluxel_rhi_core::api::Result;
use std::{
    future::Future,
    pin::Pin,
    sync::{Arc, Mutex},
    task::{Context, Poll, Waker},
};
struct State<T> {
    result: Option<Result<T>>,
    waker: Option<Waker>,
    cancelled: bool,
    completed: bool,
}
/// Observation of one native cold-object creation, cancelled independently of native work.
pub struct ColdFuture<T>(Arc<Mutex<State<T>>>);
pub(crate) struct Completer<T>(Arc<Mutex<State<T>>>);
pub(crate) fn pending<T>() -> (ColdFuture<T>, Completer<T>) {
    let state = Arc::new(Mutex::new(State {
        result: None,
        waker: None,
        cancelled: false,
        completed: false,
    }));
    (ColdFuture(state.clone()), Completer(state))
}
impl<T> ColdFuture<T> {
    pub(crate) fn ready(result: Result<T>) -> Self {
        Self(Arc::new(Mutex::new(State {
            result: Some(result),
            waker: None,
            cancelled: false,
            completed: true,
        })))
    }
}
impl<T> Completer<T> {
    pub(crate) fn finish(&self, result: Result<T>) {
        let waker = {
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            if state.cancelled || state.completed {
                return;
            }
            state.completed = true;
            state.result = Some(result);
            state.waker.take()
        };
        if let Some(waker) = waker {
            // User executor code must not unwind through an Objective-C block.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| waker.wake()));
        }
    }
}
impl<T> Future for ColdFuture<T> {
    type Output = Result<T>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<T>> {
        let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(result) = state.result.take() {
            Poll::Ready(result)
        } else {
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}
impl<T> Drop for ColdFuture<T> {
    fn drop(&mut self) {
        let (result, waker) = {
            let mut state = self.0.lock().unwrap_or_else(|e| e.into_inner());
            state.cancelled = true;
            (state.result.take(), state.waker.take())
        };
        // Native object and executor drops run outside the state mutex.
        drop(result);
        drop(waker);
    }
}
