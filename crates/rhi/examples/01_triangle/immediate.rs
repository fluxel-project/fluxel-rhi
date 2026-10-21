//! Immediate WGL workload for the indexed-triangle example.
//!
//! This module selects the GL context and uses the shared execution facade.
//! The sibling owns resources, pipeline construction, and draw calls.

use fluxel_rhi::api::*;

use super::{Triangle, utils};

fn issue_gl_frame(
    device: &Device<fluxel_rhi::gl::Gl>,
    lane: LaneId,
    triangle: &Triangle<fluxel_rhi::gl::Gl>,
    image: SurfaceImage<fluxel_rhi::gl::Gl>,
    extent: [u32; 2],
) -> Result<()> {
    let (_, outcome) = utils::execute_frame(
        device,
        lane,
        image,
        CompletionRequest::None,
        |encoder, image| triangle.encode(encoder, image, extent),
    )?;
    utils::validate_execution_outcome(&outcome)
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
    let scope = utils::GpuScope::new(&device);
    for _ in 0..frames {
        let image = surface.acquire().await?;
        issue_gl_frame(&device, lane, &triangle, image, extent)?;
    }
    scope.finish().await;
    Ok(())
}
