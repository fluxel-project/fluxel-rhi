//! Completion callbacks for explicit event failure release and sparse handoff.
//! These callbacks observe declared work; they do not infer hazards or schedule passes.
#[cfg(target_vendor = "apple")]
pub(super) struct ReleaseOnFailure {
    pub(super) completion: super::super::submit::Completion,
    pub(super) event:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>>,
}
#[cfg(target_vendor = "apple")]
impl ReleaseOnFailure {
    pub(super) fn observe(self: &std::sync::Arc<Self>) {
        use objc2_metal::MTLSharedEvent as _;
        let waker = std::task::Waker::from(self.clone());
        if matches!(self.completion.watch(&waker), Some(Err(_))) {
            // A failed producer still releases native waits. Their host
            // completion also joins the producer failure, so no result is
            // reported as successful. GPU signals use the larger value 2.
            self.event.setSignaledValue(1);
        }
    }
}
#[cfg(target_vendor = "apple")]
impl std::task::Wake for ReleaseOnFailure {
    fn wake(self: std::sync::Arc<Self>) {
        self.observe();
    }
    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        self.observe();
    }
}

#[cfg(target_vendor = "apple")]
pub(super) struct SparseDispatch {
    pub(super) ready: super::super::submit::Completion,
    pub(super) done:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>>,
    pub(super) queue: std::sync::Arc<
        std::sync::Mutex<
            objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTL4CommandQueue>>,
        >,
    >,
    pub(super) binds: std::sync::Mutex<Option<Vec<super::super::sparse::PreparedSparseBind>>>,
}
// SAFETY: this one-shot callback owns retained native resources. Access to the
// mapping queue is serialized by its shared mutex, each bind's mutable mapping
// bookkeeping is separately locked, and the event supports cross-thread signals.
// No Objective-C encoder is shared; GPU dependencies are explicit.
#[cfg(target_vendor = "apple")]
unsafe impl Send for SparseDispatch {}
#[cfg(target_vendor = "apple")]
unsafe impl Sync for SparseDispatch {}
#[cfg(target_vendor = "apple")]
impl SparseDispatch {
    pub(super) fn observe(self: &std::sync::Arc<Self>) {
        use objc2_metal::{MTL4CommandQueue as _, MTLSharedEvent as _};
        let waker = std::task::Waker::from(self.clone());
        let Some(result) = self.ready.watch(&waker) else {
            return;
        };
        let Some(binds) = self.binds.lock().unwrap_or_else(|e| e.into_inner()).take() else {
            return;
        };
        if result.is_err() {
            self.done.setSignaledValue(1);
            return;
        }
        let queue = self.queue.lock().unwrap_or_else(|e| e.into_inner());
        for bind in &binds {
            bind.enqueue(&queue);
        }
        let retained_queue = queue.clone();
        let event = self.done.clone();
        let retained_event = event.clone();
        let callback = block2::RcBlock::new(
            move |_: std::ptr::NonNull<
                objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
            >,
                  _: u64| {
                let _ = (&binds, &retained_queue, &retained_event);
            },
        );
        let listener = objc2_metal::MTLSharedEventListener::new();
        // SAFETY: Metal copies the callback; it retains mappings and the queue until signal completion.
        unsafe {
            event.notifyListener_atValue_block(
                &listener,
                1,
                block2::RcBlock::as_ptr(&callback).cast(),
            )
        };
        let native = objc2::runtime::ProtocolObject::<dyn objc2_metal::MTLEvent>::from_ref(&*event);
        queue.signalEvent_value(native, 1);
    }
}
#[cfg(target_vendor = "apple")]
impl std::task::Wake for SparseDispatch {
    fn wake(self: std::sync::Arc<Self>) {
        self.observe();
    }
    fn wake_by_ref(self: &std::sync::Arc<Self>) {
        self.observe();
    }
}
