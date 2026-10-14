//! WGL immediate route for the textured-quad example.
use super::{TextureExample, TexturePixels, data::load_ktx, utils};
use fluxel_rhi::api::*;
use std::{cell::RefCell, rc::Rc, sync::Arc};
const WGSL: &[u8] = include_bytes!("../assets/02_texture/texture.wgsl");

static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
fn err(x: ImmediateRunOutcome<fluxel_rhi::gl::Gl>) -> Error {
    match x {
        ImmediateRunOutcome::Issued(_) => Error::InvalidState,
        ImmediateRunOutcome::Failed { error, .. }
        | ImmediateRunOutcome::Indeterminate { error, .. } => error,
    }
}
fn frame(
    d: &Device<fluxel_rhi::gl::Gl>,
    lane: LaneId,
    e: &TextureExample<fluxel_rhi::gl::Gl>,
    image: SurfaceImage<fluxel_rhi::gl::Gl>,
    extent: [u32; 2],
    p: TexturePixels<'_>,
) -> Result<()> {
    let n = NEXT
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |x| x.checked_add(1),
        )
        .map_err(|_| Error::OutOfMemory)?;
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: n,
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
    let mut s = unsafe { d.immediate(plan, CompletionRequest::None)? };
    let mut c = s.begin_commands().map_err(err)?;
    e.issue(c.commands(), &image, extent, p)?;
    match c.close() {
        ImmediateRunOutcome::Issued(_) => {}
        x => return Err(err(x)),
    };
    match s.present(PresentBinding::new(image)) {
        ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
        ImmediatePresentResult::Run(x) => return Err(err(x)),
        ImmediatePresentResult::Rejected { error, .. } => return Err(error),
    };
    match s.finish() {
        ImmediateFinishOutcome::Issued(_) => Ok(()),
        ImmediateFinishOutcome::Failed(x) => Err(err(x)),
        ImmediateFinishOutcome::AlreadyReported => Err(Error::InvalidState),
    }
}
#[cfg(windows)]
pub(super) fn run() -> Result<()> {
    let (rgba, w, h) = load_ktx()?;
    let cfg = SurfaceConfig {
        extent: [1280, 720],
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    let ex = Rc::new(RefCell::new(None));
    let drop_ex = ex.clone();
    utils::gl_api::run_wgl(
        "Textured quad",
        cfg.clone(),
        move || *drop_ex.borrow_mut() = None,
        move |d, _, image, extent| {
            let lane = d
                .lanes()
                .iter()
                .find(|x| x.class == LaneClass::Graphics)
                .map(|x| x.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            let p = TexturePixels {
                rgba: &rgba,
                width: w,
                height: h,
            };
            if ex.borrow().is_none() {
                *ex.borrow_mut() = Some(utils::block_on(TextureExample::new(
                    d,
                    lane,
                    cfg.format,
                    p,
                    Some((WGSL, WGSL)),
                ))?)
            };
            frame(
                d,
                lane,
                ex.borrow().as_ref().expect("initialized"),
                image,
                extent,
                p,
            )
        },
    )
}

/// Keeps the canvas owner, device and all sampled resources in one browser
/// context generation while each frame is issued and presented immediately.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) async fn run_browser(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
    use fluxel_rhi::gl::GlInstance;

    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    let (rgba, width, height) = load_ktx()?;
    let pixels = TexturePixels {
        rgba: &rgba,
        width,
        height,
    };
    let extent = [canvas.width(), canvas.height()];
    if extent.contains(&0) {
        return Err(Error::InvalidRange);
    }
    let native = GlInstance::from_webgl2_canvas(canvas)?;
    // SAFETY: the browser context owner and canvas outlive every device,
    // resource, surface lease and immediate session in this function.
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
    let example = TextureExample::new(&device, lane, format, pixels, Some((WGSL, WGSL))).await?;
    for _ in 0..frames {
        let image = surface.acquire().await?;
        frame(&device, lane, &example, image, extent, pixels)?;
    }
    Ok(())
}
