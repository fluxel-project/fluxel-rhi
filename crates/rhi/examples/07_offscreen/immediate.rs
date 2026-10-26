//! Immediate WGL adapter for the shared offscreen mirror workload.
//!
//! The workload issues its offscreen pass, explicit texture transitions, and
//! surface pass through one owner-thread command encoder. GL presentation is
//! the next immediate step; this adapter does not create deferred work.

use std::cell::RefCell;
use std::rc::Rc;

use fluxel_rhi::api::*;
use fluxel_rhi::backend::*;
use fluxel_rhi::gl::Gl;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
use fluxel_rhi::gl::GlInstance;

use super::{Offscreen, utils};

fn issue_frame(
    device: &Device<Gl>,
    lane: LaneId,
    workload: &mut Offscreen<Gl>,
    image: SurfaceImage<Gl>,
    extent: [u32; 2],
    time: f32,
) -> Result<()> {
    utils::execute_frame(
        device,
        lane,
        image,
        CompletionRequest::None,
        |commands, image| workload.encode(commands, device, image, extent, time),
    )?;
    workload.commit_frame();
    Ok(())
}

#[cfg(any(windows, all(target_os = "linux", feature = "native-gles-egl")))]
pub(super) fn run_gl() -> Result<()> {
    let config = SurfaceConfig {
        extent: [1280, 720],
        // WGL exposes the tested RGBA default framebuffer format; the scene's
        // visible color semantics are identical to the BGRA native routes.
        format: TextureFormat::Rgba8Unorm,
        present_mode: PresentMode::Fifo,
        alpha_mode: CompositeAlphaMode::Opaque,
        color_space: ColorSpace::Srgb,
        usage: SurfaceUsage::RENDER_ATTACHMENT,
        maximum_frame_latency: 1,
    };
    let start = std::time::Instant::now();
    let workload = Rc::new(RefCell::new(None));
    let dropped_workload = Rc::clone(&workload);
    utils::gl_api::run_native(
        "Offscreen mirror rendering",
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
                *workload.borrow_mut() = Some(utils::block_on(Offscreen::<Gl>::new(
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

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) async fn run_webgl2(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    let extent = [canvas.width(), canvas.height()];
    if extent.contains(&0) {
        return Err(Error::SurfaceLost);
    }
    let native = GlInstance::from_webgl2_canvas(canvas)?;
    // SAFETY: the canvas-backed owner remains retained by the instance,
    // device, and surface while all immediate GL calls run on this browser thread.
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
    let mut workload = Offscreen::<Gl>::new(&device, lane, format, extent).await?;
    let scope = utils::GpuScope::new(&device);
    for frame in 0..frames {
        let image = surface.acquire().await?;
        issue_frame(
            &device,
            lane,
            &mut workload,
            image,
            extent,
            frame as f32 / 60.0,
        )?;
    }
    scope.finish().await;
    Ok(())
}
