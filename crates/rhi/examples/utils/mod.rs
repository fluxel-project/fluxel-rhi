//! Shared RHI example harness.
//!
//! A native host selects one concrete backend at process initialization. The
//! examples stay generic over that backend, so resources and commands retain
//! static dispatch in every hot path.

use std::future::Future;
use std::sync::Arc;

use fluxel_rhi::api::{
    Backend, CompletionPoint, Device, Error, ImmediateBackend, ImmediateFinishOutcome,
    ImmediateRunOutcome, Result, Shader, ShaderBackend, ShaderDesc, ShaderEntryPoint, ShaderFormat,
    ShaderRoute, ShaderStage, Surface, SurfaceConfig, SurfaceImage,
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

/// Native example host with ordered input callbacks. The callback receives
/// input events before the next redraw and may share state with `draw`.
#[cfg(windows)]
pub trait ExampleInputHost<B: Backend> {
    fn run_with_input(
        self,
        title: &str,
        configure: SurfaceConfig,
        draw: impl FnMut(&Device<B>, &mut Surface<B>, SurfaceImage<B>, [u32; 2]) -> Result<()> + 'static,
        input: impl FnMut(&fluxel_host::WindowEvent) + 'static,
    ) -> Result<()>;
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

#[path = "submission.rs"]
mod submission;
pub use submission::{
    FrameCompletion, FrameSubmission, submit_and_present, submit_and_present_async,
    submit_and_present_point, submit_and_present_point_async, submit_frame,
};
#[path = "lifecycle.rs"]
mod lifecycle;
pub use lifecycle::GpuScope;
#[path = "execution.rs"]
mod execution;
pub use execution::{execute_frame, execution_end_checkpoint, validate_execution_outcome};
