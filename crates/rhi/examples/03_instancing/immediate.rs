//! Immediate GL routes for the instanced rock-field example.
//!
//! WGL and WebGL2 use the shared execution facade. Commands issue native calls
//! as they run; shaders remain the checked-in WebGPU WGSL assets.

use fluxel_rhi::api::*;

use super::Instancing;

fn issue_frame(
    device: &Device<fluxel_rhi::gl::Gl>,
    lane: LaneId,
    scene: &mut Instancing<fluxel_rhi::gl::Gl>,
    image: SurfaceImage<fluxel_rhi::gl::Gl>,
    extent: [u32; 2],
    time: f32,
) -> Result<()> {
    let (_, outcome) = super::utils::execute_frame(
        device,
        lane,
        image,
        CompletionRequest::None,
        |encoder, image| scene.issue(encoder, image, extent, time),
    )?;
    super::utils::validate_execution_outcome(&outcome)?;
    scene.mark_submitted();
    Ok(())
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

#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    all(target_os = "linux", feature = "native-gles-egl")
))]
pub(super) fn run_native() -> Result<()> {
    let config = config();
    let scene = std::rc::Rc::new(std::cell::RefCell::new(None));
    let dropped_scene = std::rc::Rc::clone(&scene);
    let started = std::time::Instant::now();
    crate::utils::gl_api::run_native(
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
    let scope = super::utils::GpuScope::new(&device);
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
    scope.finish().await;
    Ok(())
}
