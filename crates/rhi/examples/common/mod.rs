//! Shared RHI example harness.
//!
//! A native host selects one concrete backend at process initialization. The
//! examples stay generic over that backend, so resources and commands retain
//! static dispatch in every hot path.

use std::future::Future;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use fluxel_rhi::api::{
    AccessMask, AccessScope, Backend, BoundStep, CompletionRequest, DeferredBackend, Device, Error,
    LaneId, PlanId, PresentBinding, RecordedWork, Result, StageMask, StepDesc, StepKind,
    StepPayload, StepWait, SubmissionPlan, SubmitOutcome, SubmitPacket, Surface, SurfaceConfig,
    SurfaceImage,
};

#[cfg(windows)]
pub mod windows_api;

static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

/// Drives a cold-path RHI future on the current native host thread.
pub fn block_on<F: Future>(future: F) -> F::Output {
    struct ThreadWaker(std::thread::Thread);
    impl std::task::Wake for ThreadWaker {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(ThreadWaker(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match Future::poll(future.as_mut(), &mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}

/// Narrow window/runtime adapter for typed native examples.
///
/// Window ownership, event loops, frame pacing and the one-time DX12 choice are
/// outer-runtime work. The adapter invokes the closure with one concrete `B`;
/// it must retain the surface lease until the closure hands it to a compiled
/// submission plan.
pub trait ExampleHost<B: Backend> {
    fn run(
        self,
        title: &str,
        configure: SurfaceConfig,
        draw: impl FnMut(&Device<B>, &mut Surface<B>, SurfaceImage<B>, [u32; 2]) -> Result<()> + 'static,
    ) -> Result<()>;
}

/// Starts one statically typed example through its native host adapter.
pub fn run_example<B: Backend, H: ExampleHost<B>>(
    host: H,
    title: &str,
    configure: SurfaceConfig,
    draw: impl FnMut(&Device<B>, &mut Surface<B>, SurfaceImage<B>, [u32; 2]) -> Result<()> + 'static,
) -> Result<()> {
    host.run(title, configure, draw)
}

/// Converts an unsupported native route to the operation the example asked
/// for, without introducing a second dynamic error hierarchy.
pub fn require<T>(value: Result<T>, what: &'static str) -> Result<T> {
    value.map_err(|error| match error {
        Error::Unsupported(_) => Error::Unsupported(what),
        other => other,
    })
}

/// Submits one recorded frame on a single queue, then presents its surface lease.
/// The example waits for the requested end checkpoint before reusing frame data.
pub fn submit_and_present<B: DeferredBackend>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    image: SurfaceImage<B>,
) -> Result<()> {
    if lane.device != device.id() || image.device() != device.id() {
        return Err(Error::WrongDevice);
    }
    let id = PlanId {
        device: device.id(),
        ordinal: NEXT_PLAN.fetch_add(1, Ordering::Relaxed),
    };
    let present_wait = StepWait {
        previous: fluxel_rhi::api::StepId { plan: id, index: 0 },
        destination: AccessScope {
            stages: StageMask::ALL_COMMANDS,
            accesses: AccessMask::ALL,
        },
    };
    // Both steps target the same ordered graphics lane. The present dependency
    // is explicitly represented; the backend resolves it by native queue order.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([
                StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane,
                    kind: StepKind::Present,
                    waits: vec![present_wait],
                },
            ]),
        )
    };
    let steps = vec![
        BoundStep {
            step: plan.step(0),
            payload: StepPayload::Commands(vec![work]),
        },
        BoundStep {
            step: plan.step(1),
            payload: StepPayload::Present(PresentBinding::new(image)),
        },
    ];
    // The host retains all demo resources and this surface until the requested
    // completion is observed. That fulfills the native submission lifetime rule.
    let outcome = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps,
            completion: CompletionRequest::End,
        })
    };
    match outcome {
        SubmitOutcome::Accepted(receipt) => {
            if receipt.accepted.len() != 2
                || receipt.presents.len() != 1
                || !receipt.presents[0].accepted
            {
                return Err(Error::InvalidState);
            }
            if receipt.checkpoints.is_empty() {
                block_on(device.wait_idle())?;
            } else {
                for checkpoint in &receipt.checkpoints {
                    block_on(device.wait_completion(&checkpoint.point))?;
                }
            }
            Ok(())
        }
        SubmitOutcome::Rejected { error, .. } => Err(error),
        SubmitOutcome::PartiallyAccepted { error, .. }
        | SubmitOutcome::Indeterminate { error, .. } => {
            // Some native work may have crossed the queue boundary. Keep the
            // demo's resources alive through device idle before surfacing it.
            block_on(device.wait_idle())?;
            Err(error)
        }
    }
}
