//! Completion wake registrations and native retirement evidence.
use super::*;
use std::sync::Arc;
use std::task::Wake;
use wasm_bindgen_test::*;
struct Noop;
impl Wake for Noop {
    fn wake(self: Arc<Self>) {}
}
fn pending() -> Completion {
    Completion(
        Rc::new(RefCell::new(CompletionInner {
            state: api::CompletionState::Pending,
            native_settled: false,
            wakers: BTreeMap::new(),
            next_registration: 0,
        })),
        None,
    )
}
#[wasm_bindgen_test]
fn dropped_completion_future_removes_its_replaced_waker() {
    let point = pending();
    let mut future = point.clone();
    let first = Waker::from(Arc::new(Noop));
    let second = Waker::from(Arc::new(Noop));
    assert!(
        Pin::new(&mut future)
            .poll(&mut Context::from_waker(&first))
            .is_pending()
    );
    assert!(
        Pin::new(&mut future)
            .poll(&mut Context::from_waker(&second))
            .is_pending()
    );
    assert_eq!(point.0.borrow().wakers.len(), 1);
    assert!(
        point
            .0
            .borrow()
            .wakers
            .values()
            .next()
            .unwrap()
            .will_wake(&second)
    );
    drop(future);
    assert!(point.0.borrow().wakers.is_empty());
}
#[wasm_bindgen_test]
fn public_failure_does_not_supply_native_retirement_evidence() {
    let point = pending();
    point.settled(api::CompletionState::Failed(Error::DeviceLost));
    assert!(!point.0.borrow().native_settled);
    assert!(matches!(point.state(), api::CompletionState::Failed(_)));
}
