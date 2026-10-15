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
    AccessMask, AccessScope, Backend, BoundStep, CompletionPoint, CompletionRequest,
    DeferredBackend, Device, Error, ImmediateBackend, ImmediateFinishOutcome, ImmediateRunOutcome,
    LaneId, PlanId, PresentBinding, RecordedWork, Result, Shader, ShaderBackend, ShaderDesc,
    ShaderEntryPoint, ShaderFormat, ShaderRoute, ShaderStage, StageMask, StepDesc, StepKind,
    StepPayload, StepWait, SubmissionPlan, SubmitOutcome, SubmitPacket, Surface, SurfaceConfig,
    SurfaceImage,
};

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
pub mod gl_api;

#[cfg(windows)]
pub mod windows_api;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal_api;

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

/// Returns the single-lane End checkpoint from a finished immediate session.
/// GL examples use this to observe requested GPU completion before dropping
/// frame-local resources or returning to the host event loop.
pub fn immediate_end_checkpoint<B: ImmediateBackend>(
    outcome: ImmediateFinishOutcome<B>,
) -> Result<CompletionPoint<B>> {
    match outcome {
        ImmediateFinishOutcome::Issued(mut prefix) if prefix.checkpoints.len() == 1 => {
            Ok(prefix.checkpoints.remove(0).point)
        }
        ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Failed { error, .. })
        | ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Indeterminate { error, .. }) => {
            Err(error)
        }
        ImmediateFinishOutcome::Issued(_)
        | ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Issued(_))
        | ImmediateFinishOutcome::AlreadyReported => Err(Error::InvalidState),
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

/// Selects a checked-in native shader at creation time using the backend's
/// declared route. Compilation errors from a selected route are returned as-is.
pub async fn native_shader<B: ShaderBackend>(
    device: &Device<B>,
    label: &str,
    entry: &str,
    stage: ShaderStage,
    dxil: &[u8],
    spirv: &[u8],
) -> Result<Shader<B>> {
    for (format, bytes) in [(ShaderFormat::Dxil, dxil), (ShaderFormat::SpirV, spirv)] {
        let desc = ShaderDesc {
            label: Some(label.into()),
            format,
            bytes: Arc::from(bytes),
            entries: vec![ShaderEntryPoint {
                name: entry.into(),
                stage,
                workgroup_size: None,
            }],
            required_features: vec![],
        };
        if !matches!(device.shader_route(&desc), ShaderRoute::Unsupported) {
            return device.shader(desc).await;
        }
    }
    Err(Error::Unsupported("DXIL or SPIR-V shader route"))
}

/// Picks one native shader artifact, including WGSL for browser WebGPU.
pub async fn native_shader_with_wgsl<B: ShaderBackend>(
    device: &Device<B>,
    label: &str,
    entry: &str,
    stage: ShaderStage,
    dxil: &[u8],
    spirv: &[u8],
    wgsl: &[u8],
) -> Result<Shader<B>> {
    for (format, bytes) in [
        (ShaderFormat::Dxil, dxil),
        (ShaderFormat::Wgsl, wgsl),
        (ShaderFormat::SpirV, spirv),
    ] {
        let desc = ShaderDesc {
            label: Some(label.into()),
            format,
            bytes: Arc::from(bytes),
            entries: vec![ShaderEntryPoint {
                name: entry.into(),
                stage,
                workgroup_size: None,
            }],
            required_features: vec![],
        };
        if !matches!(device.shader_route(&desc), ShaderRoute::Unsupported) {
            return device.shader(desc).await;
        }
    }
    Err(Error::Unsupported("DXIL, SPIR-V, or WGSL shader route"))
}

/// Submits one recorded frame on a single queue, then presents its surface lease.
/// The example waits for the requested end checkpoint before reusing frame data.
pub fn submit_and_present<B: DeferredBackend>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    image: SurfaceImage<B>,
) -> Result<()> {
    let outcome = issue_submit_and_present(device, lane, work, image)?;
    match outcome {
        SubmitOutcome::Accepted(receipt) => {
            validate_present_receipt(&receipt)?;
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
            block_on(device.wait_idle())?;
            Err(error)
        }
    }
}

/// Browser hosts await completion without blocking their event loop.
#[cfg(target_arch = "wasm32")]
pub async fn submit_and_present_async<B: DeferredBackend>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    image: SurfaceImage<B>,
) -> Result<()> {
    let outcome = issue_submit_and_present(device, lane, work, image)?;
    match outcome {
        SubmitOutcome::Accepted(receipt) => {
            validate_present_receipt(&receipt)?;
            if receipt.checkpoints.is_empty() {
                device.wait_idle().await?;
            } else {
                for checkpoint in &receipt.checkpoints {
                    device.wait_completion(&checkpoint.point).await?;
                }
            }
            Ok(())
        }
        SubmitOutcome::Rejected { error, .. } => Err(error),
        SubmitOutcome::PartiallyAccepted { error, .. }
        | SubmitOutcome::Indeterminate { error, .. } => {
            device.wait_idle().await?;
            Err(error)
        }
    }
}

fn validate_present_receipt<B: DeferredBackend>(
    receipt: &fluxel_rhi::api::SubmissionReceipt<B>,
) -> Result<()> {
    if receipt.accepted.len() != 2 || receipt.presents.len() != 1 || !receipt.presents[0].accepted {
        return Err(Error::InvalidState);
    }
    Ok(())
}

fn issue_submit_and_present<B: DeferredBackend>(
    device: &Device<B>,
    lane: LaneId,
    work: RecordedWork<B>,
    image: SurfaceImage<B>,
) -> Result<SubmitOutcome<B>> {
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
    // SAFETY: both steps target the same ordered graphics lane. The present
    // dependency names this plan's command step, making the compiled order
    // realizable by native queue order without backend DAG reconstruction.
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
    // SAFETY: the host retains resources referenced by `work` and the image
    // lease until the end checkpoint completes. Both bound steps belong to
    // this plan and device, fulfilling the native submission contract.
    Ok(unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps,
            completion: CompletionRequest::End,
        })
    })
}
