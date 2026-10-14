//! Immediate GL routes for the instanced rock-field example.
//!
//! Both WGL and WebGL2 create one immediate session per acquired surface
//! image. The shared workload issues native calls as it runs and keeps no Rust
//! command replay list; its shaders remain the checked-in WebGPU WGSL assets.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use fluxel_rhi::api::*;

use super::Instancing;

static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

fn immediate_failure(outcome: ImmediateRunOutcome<fluxel_rhi::gl::Gl>) -> Error {
    match outcome {
        ImmediateRunOutcome::Issued(_) => Error::InvalidState,
        ImmediateRunOutcome::Failed { error, .. }
        | ImmediateRunOutcome::Indeterminate { error, .. } => error,
    }
}

fn issue_frame(
    device: &Device<fluxel_rhi::gl::Gl>,
    lane: LaneId,
    scene: &mut Instancing<fluxel_rhi::gl::Gl>,
    image: SurfaceImage<fluxel_rhi::gl::Gl>,
    extent: [u32; 2],
    time: f32,
) -> Result<()> {
    let ordinal = NEXT_PLAN
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| Error::OutOfMemory)?;
    // SAFETY: this is a precompiled, one-lane command/present order. The
    // scene retains every resource until the host tears down its context.
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
    // SAFETY: this function closes the only command step, consumes the
    // acquired lease exactly once, and finishes before its borrowed scene can
    // be released. Completion is intentionally omitted because the sample
    // never reuses or retires frame-local storage.
    let mut session = unsafe { device.immediate(plan, CompletionRequest::None)? };
    let mut immediate = session.begin_commands().map_err(immediate_failure)?;
    scene.issue(immediate.commands(), &image, extent, time)?;
    match immediate.close() {
        ImmediateRunOutcome::Issued(_) => {}
        other => return Err(immediate_failure(other)),
    }
    match session.present(PresentBinding::new(image)) {
        ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
        ImmediatePresentResult::Run(other) => return Err(immediate_failure(other)),
        ImmediatePresentResult::Rejected { error, .. } => return Err(error),
    }
    match session.finish() {
        ImmediateFinishOutcome::Issued(_) => {
            scene.mark_submitted();
            Ok(())
        }
        ImmediateFinishOutcome::Failed(outcome) => Err(immediate_failure(outcome)),
        ImmediateFinishOutcome::AlreadyReported => Err(Error::InvalidState),
    }
}

fn graphics_lane(device: &Device<fluxel_rhi::gl::Gl>) -> Result<LaneId> {
    device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .map(|lane| lane.id)
        .ok_or(Error::Unsupported("GL graphics lane"))
}

fn config() -> SurfaceConfig {
    SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    }
}

#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(super) fn run_wgl() -> Result<()> {
    let config = config();
    let scene = std::rc::Rc::new(std::cell::RefCell::new(None));
    let dropped_scene = std::rc::Rc::clone(&scene);
    let started = std::time::Instant::now();
    crate::utils::gl_api::run_wgl(
        "Instanced rock field",
        config.clone(),
        move || *dropped_scene.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = graphics_lane(device)?;
            if scene.borrow().is_none() {
                *scene.borrow_mut() = Some(crate::utils::block_on(Instancing::new(
                    device,
                    lane,
                    config.format,
                    extent,
                ))?);
            }
            let mut scene = scene.borrow_mut();
            let scene = scene.as_mut().ok_or(Error::InvalidState)?;
            issue_frame(
                device,
                lane,
                scene,
                image,
                extent,
                started.elapsed().as_secs_f32(),
            )
        },
    )
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) async fn run_webgl2(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    use fluxel_rhi::gl::{Gl, GlInstance};

    let extent = [canvas.width(), canvas.height()];
    let native = GlInstance::from_webgl2_canvas(canvas)?;
    // SAFETY: the browser canvas context remains owned by this async function;
    // all GL calls occur on its JavaScript owner thread before it returns.
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
    let lane = graphics_lane(&device)?;
    let mut surface = Gl::surface(&device)?;
    let mut config = config();
    config.extent = extent;
    surface.configure(&config).await?;
    let mut scene = Instancing::new(&device, lane, config.format, extent).await?;
    for frame in 0..frames {
        let image = surface.acquire().await?;
        issue_frame(
            &device,
            lane,
            &mut scene,
            image,
            extent,
            frame as f32 / 60.0,
        )?;
    }
    Ok(())
}
