//! Immediate WGL adapter for the shared indexed-indirect plant workload.
//!
//! GL 4.x issues each of the workload's indirect records on the context owner
//! thread. Multi-draw remains unsupported, so the shared workload selects its
//! existing single-draw fallback without reading or replaying argument bytes.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use fluxel_rhi::api::*;
use fluxel_rhi::gl::Gl;

use super::{utils, workload::Workload};

static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

fn immediate_error(outcome: ImmediateRunOutcome<Gl>) -> Error {
    match outcome {
        ImmediateRunOutcome::Issued(_) => Error::InvalidState,
        ImmediateRunOutcome::Failed { error, .. }
        | ImmediateRunOutcome::Indeterminate { error, .. } => error,
    }
}

/// Runs the visual fallback on a caller-owned WebGL2 canvas. Argument bytes
/// are decoded by `Workload::encode` and issued as direct indexed draws; this
/// path does not record a CPU command list or claim indirect-draw support.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) async fn run_webgl2(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
    use fluxel_rhi::gl::GlInstance;

    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    let extent = [canvas.width(), canvas.height()];
    if extent.contains(&0) {
        return Err(Error::SurfaceLost);
    }
    let native = GlInstance::from_webgl2_canvas(canvas)?;
    // SAFETY: all browser GL calls stay on this future's JavaScript owner
    // thread, and the instance/device retain the canvas context owner.
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
    if device.capabilities().indirect_draw {
        return Err(Error::InvalidState);
    }
    let format = TextureFormat::Rgba8Unorm;
    let mut surface = Gl::surface(&device)?;
    surface
        .configure(&SurfaceConfig {
            extent,
            format,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 1,
        })
        .await?;
    let mut workload = Workload::<Gl>::new(&device, lane, format, extent).await?;
    for frame in 0..frames {
        let image = surface.acquire().await?;
        let ordinal = NEXT_PLAN
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| Error::OutOfMemory)?;
        // SAFETY: command and present steps use the same immediate GL lane.
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
        // SAFETY: the leased image and workload live until commands close and
        // the following Present step consumes this exact lease.
        let mut session = unsafe { device.immediate(plan, CompletionRequest::End)? };
        let mut commands = session.begin_commands().map_err(immediate_error)?;
        let time = web_sys::window()
            .and_then(|window| window.performance())
            .map(|performance| (performance.now() / 1000.0) as f32)
            .unwrap_or(frame as f32 / 60.0);
        let commit = workload.encode(commands.commands(), &device, &image, extent, time)?;
        match commands.close() {
            ImmediateRunOutcome::Issued(_) => workload.commit_frame(commit),
            outcome => return Err(immediate_error(outcome)),
        }
        match session.present(PresentBinding::new(image)) {
            ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
            ImmediatePresentResult::Run(outcome) => return Err(immediate_error(outcome)),
            ImmediatePresentResult::Rejected { error, .. } => return Err(error),
        }
        let prefix = match session.finish() {
            ImmediateFinishOutcome::Issued(prefix) => prefix,
            ImmediateFinishOutcome::Failed(outcome) => return Err(immediate_error(outcome)),
            ImmediateFinishOutcome::AlreadyReported => return Err(Error::InvalidState),
        };
        let point = prefix
            .checkpoints
            .into_iter()
            .next()
            .map(|checkpoint| checkpoint.point)
            .ok_or(Error::InvalidState)?;
        device.wait_completion(&point).await?;
    }
    Ok(())
}

fn issue_frame(
    device: &Device<Gl>,
    lane: LaneId,
    workload: &mut Workload<Gl>,
    image: SurfaceImage<Gl>,
    extent: [u32; 2],
    time: f32,
) -> Result<()> {
    let ordinal = NEXT_PLAN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::OutOfMemory)?;
    // SAFETY: both steps target the same adopted GL lane and execute in order.
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
    // SAFETY: the current owner-thread callback immediately issues Commands;
    // the acquired image remains leased until the following Present step.
    let mut session = unsafe { device.immediate(plan, CompletionRequest::None)? };
    let mut commands = session.begin_commands().map_err(immediate_error)?;
    let commit = workload.encode(commands.commands(), device, &image, extent, time)?;
    match commands.close() {
        ImmediateRunOutcome::Issued(_) => workload.commit_frame(commit),
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
        // GL uses the caller-owned default framebuffer, whose surface contract
        // admits one outstanding lease on its immediate context.
        maximum_frame_latency: 1,
    };
    let start = std::time::Instant::now();
    let workload = Rc::new(RefCell::new(None));
    let dropped_workload = Rc::clone(&workload);
    utils::gl_api::run_native(
        "Indirect draw",
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
                *workload.borrow_mut() = Some(utils::block_on(Workload::<Gl>::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            issue_frame(
                device,
                lane,
                workload.borrow_mut().as_mut().ok_or(Error::InvalidState)?,
                image,
                extent,
                start.elapsed().as_secs_f32(),
            )
        },
    )
}

#[cfg(not(any(windows, all(target_os = "linux", feature = "native-gles-egl"))))]
pub(super) fn run() -> Result<()> {
    Err(Error::Unsupported("10 GL WGL host"))
}
