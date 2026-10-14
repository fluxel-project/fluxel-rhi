//! Immediate WGL execution for the dynamic-uniform cube grid.
//!
//! Resource and pipeline construction stays in the parent module so the GL
//! route uses the same WGSL source, buffers, bind-group layout, and draw
//! workload as deferred backends.  This module owns only the immediate-session
//! skeleton: GL calls are issued while the session is open and presentation
//! hands off the acquired default-framebuffer lease.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
#[cfg(any(windows, all(target_os = "linux", feature = "native-gles-egl")))]
use std::{cell::RefCell, rc::Rc};

use fluxel_rhi::api::*;

use super::DynamicUniforms;

static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

fn immediate_error(outcome: ImmediateRunOutcome<fluxel_rhi::gl::Gl>) -> Error {
    match outcome {
        ImmediateRunOutcome::Issued(_) => Error::InvalidState,
        ImmediateRunOutcome::Failed { error, .. }
        | ImmediateRunOutcome::Indeterminate { error, .. } => error,
    }
}

fn issue_frame(
    device: &Device<fluxel_rhi::gl::Gl>,
    lane: LaneId,
    workload: &DynamicUniforms<fluxel_rhi::gl::Gl>,
    image: SurfaceImage<fluxel_rhi::gl::Gl>,
    extent: [u32; 2],
    time: f32,
) -> Result<()> {
    let ordinal = NEXT_PLAN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::OutOfMemory)?;
    // SAFETY: the direct command then present steps are fully ordered on the
    // one GL lane. This closure completes both steps before returning, and no
    // command is retained for later execution.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal,
            },
            Arc::from([
                StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane,
                    kind: StepKind::Present,
                    waits: vec![],
                },
            ]),
        )
    };
    // SAFETY: `issue_frame` closes the sole encoder and consumes `image` via
    // the matching present step before it returns, satisfying the compiled
    // plan's immediate-session ownership contract.
    let mut session = unsafe { device.immediate(plan, CompletionRequest::None)? };
    let mut commands = session.begin_commands().map_err(immediate_error)?;
    workload.encode(commands.commands(), &image, extent, time)?;
    match commands.close() {
        ImmediateRunOutcome::Issued(_) => {}
        outcome => return Err(immediate_error(outcome)),
    }
    match session.present(PresentBinding::new(image)) {
        ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
        ImmediatePresentResult::Run(outcome) => return Err(immediate_error(outcome)),
        ImmediatePresentResult::Rejected { error, .. } => return Err(error),
    }
    match session.finish() {
        ImmediateFinishOutcome::Issued(_) => Ok(()),
        ImmediateFinishOutcome::Failed(outcome) => Err(immediate_error(outcome)),
        ImmediateFinishOutcome::AlreadyReported => Err(Error::InvalidState),
    }
}

#[cfg(any(windows, all(target_os = "linux", feature = "native-gles-egl")))]
pub(super) fn run() -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    let workload = Rc::new(RefCell::new(None));
    let dropped_workload = Rc::clone(&workload);
    let started = std::time::Instant::now();
    super::utils::gl_api::run_native(
        "Dynamic uniform buffers",
        config.clone(),
        move || *dropped_workload.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            if workload.borrow().is_none() {
                *workload.borrow_mut() = Some(super::utils::block_on(DynamicUniforms::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            issue_frame(
                device,
                lane,
                workload.borrow().as_ref().ok_or(Error::InvalidState)?,
                image,
                extent,
                started.elapsed().as_secs_f32(),
            )
        },
    )
}

/// Runs the same direct workload against a caller-owned WebGL2 canvas.
///
/// WebGL has no native swap call after command issuance. The immediate
/// session's present step hands off the Core lease and lets the backend flush
/// the canvas context, preserving the same ownership boundary as WGL.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) async fn run_webgl2(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    use fluxel_rhi::gl::{Gl, GlInstance};

    let native = GlInstance::from_webgl2_canvas(canvas.clone())?;
    // SAFETY: `native` is a newly-created browser context owner. It remains
    // retained by `instance`, then by the opened typed device and surface for
    // this function's whole async execution on the browser owner thread.
    let instance = unsafe { Instance::<Gl>::from_native(InstanceId::allocate()?, native) };
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or(Error::Unsupported("WebGL2 adapter"))?;
    let device = instance
        .open(DeviceRequest {
            adapter: adapter.id,
            required_features: vec![],
        })
        .await?;
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .map(|lane| lane.id)
        .ok_or(Error::Unsupported("WebGL2 graphics lane"))?;
    let extent = [canvas.width(), canvas.height()];
    if extent.contains(&0) {
        return Err(Error::SurfaceLost);
    }
    let config = SurfaceConfig {
        extent,
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    let mut surface = Gl::surface(&device)?;
    surface.configure(&config).await?;
    let workload = DynamicUniforms::new(&device, lane, config.format, extent).await?;
    for frame in 0..frames {
        let image = surface.acquire().await?;
        issue_frame(&device, lane, &workload, image, extent, frame as f32 / 60.0)?;
    }
    Ok(())
}
