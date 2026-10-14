//! Immediate WGL workload for the indexed-triangle example.
//!
//! This module owns only GL-specific direct-session issuance. The sibling
//! owns resources, pipeline construction, and the shared draw calls.

use fluxel_rhi::api::*;

use super::{Triangle, utils};

static NEXT_PLAN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn immediate_failure(outcome: ImmediateRunOutcome<fluxel_rhi::gl::Gl>) -> Error {
    match outcome {
        ImmediateRunOutcome::Issued(_) => Error::InvalidState,
        ImmediateRunOutcome::Failed { error, .. }
        | ImmediateRunOutcome::Indeterminate { error, .. } => error,
    }
}

fn issue_gl_frame(
    device: &Device<fluxel_rhi::gl::Gl>,
    lane: LaneId,
    triangle: &Triangle<fluxel_rhi::gl::Gl>,
    image: SurfaceImage<fluxel_rhi::gl::Gl>,
    extent: [u32; 2],
) -> Result<()> {
    let plan_ordinal = NEXT_PLAN
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |ordinal| ordinal.checked_add(1),
        )
        .map_err(|_| Error::OutOfMemory)?;
    // SAFETY: this one-context plan has an already compiled direct order. The
    // closure retains the triangle and its buffers until each session has
    // completed its present handoff; `None` intentionally creates no
    // per-frame host fence because the sample never reclaims those resources.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: plan_ordinal,
            },
            std::sync::Arc::from([
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
    // SAFETY: this function explicitly closes the only command step, presents
    // its exclusive lease, and finishes before returning. The captured
    // triangle outlives every issued GL call for the lifetime of the host.
    let mut session = unsafe { device.immediate(plan, CompletionRequest::None)? };
    let mut immediate = session.begin_commands().map_err(immediate_failure)?;
    triangle.encode(immediate.commands(), &image, extent)?;
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
        ImmediateFinishOutcome::Issued(_) => Ok(()),
        ImmediateFinishOutcome::Failed(outcome) => Err(immediate_failure(outcome)),
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
    let triangle = std::rc::Rc::new(std::cell::RefCell::new(None));
    let dropped_triangle = std::rc::Rc::clone(&triangle);
    utils::gl_api::run_native(
        "Basic indexed triangle",
        config.clone(),
        move || *dropped_triangle.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            let mut triangle = triangle.borrow_mut();
            if triangle.is_none() {
                *triangle = Some(utils::block_on(Triangle::new(device, lane, config.format))?);
            }
            issue_gl_frame(
                device,
                lane,
                triangle.as_ref().expect("initialized above"),
                image,
                extent,
            )
        },
    )
}

/// Uses the same immediate workload on a browser-owned WebGL2 canvas.
/// The browser owner retains the context and canvas until every typed GPU
/// resource, surface lease and completion point has been dropped.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) async fn run_browser(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
    use fluxel_rhi::gl::GlInstance;

    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    let extent = [canvas.width(), canvas.height()];
    if extent.contains(&0) {
        return Err(Error::InvalidRange);
    }
    let native = GlInstance::from_webgl2_canvas(canvas)?;
    // SAFETY: the browser owner retains the context and its generation for
    // this instance; all resources and leases below are dropped before it.
    let instance =
        unsafe { Instance::<fluxel_rhi::gl::Gl>::from_native(InstanceId::allocate()?, native) };
    let adapter = instance
        .enumerate()?
        .into_iter()
        .next()
        .ok_or(Error::Unsupported("WebGL2 adapter"))?;
    let device = instance
        .open(DeviceRequest {
            adapter: adapter.id,
            required_features: Vec::new(),
        })
        .await?;
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .map(|lane| lane.id)
        .ok_or(Error::Unsupported("WebGL2 graphics lane"))?;
    let mut surface = fluxel_rhi::gl::Gl::surface(&device)?;
    let format = *surface
        .capabilities()?
        .formats
        .first()
        .ok_or(Error::Unsupported("WebGL2 surface format"))?;
    let config = SurfaceConfig {
        extent,
        format,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    surface.configure(&config).await?;
    let triangle = Triangle::new(&device, lane, format).await?;
    for _ in 0..frames {
        let image = surface.acquire().await?;
        issue_gl_frame(&device, lane, &triangle, image, extent)?;
    }
    Ok(())
}
