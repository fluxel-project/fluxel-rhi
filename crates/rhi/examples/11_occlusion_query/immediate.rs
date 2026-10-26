//! Immediate WGL adapter for the shared occlusion-query workload.
//!
//! The query result is resolved by GL's blocking result query before the
//! frame's host completion fence. This preserves the Core readback contract
//! without turning immediate commands into a deferred command list.

use std::cell::RefCell;
use std::rc::Rc;

use fluxel_rhi::api::*;
use fluxel_rhi::backend::*;
use fluxel_rhi::gl::Gl;

use super::{OcclusionWorkloadBackend, Workload, utils};

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) async fn run_webgl2(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
    use fluxel_rhi::gl::GlInstance;

    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    let extent = [canvas.width(), canvas.height()];
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
    if device.capabilities().occlusion {
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
    let mut workload = Workload::<Gl>::new_visual_fallback(&device, lane, format).await?;
    let scope = utils::GpuScope::new(&device);
    for _ in 0..frames {
        let image = surface.acquire().await?;
        let (_, outcome) = utils::execute_frame(
            &device,
            lane,
            image,
            CompletionRequest::End,
            |encoder, image| {
                let ticket = workload.encode_scene(&device, encoder, image, extent)?;
                let unexpected = ticket.is_some();
                workload.retain_ticket(ticket)?;
                if unexpected {
                    return Err(Error::InvalidState);
                }
                Ok(())
            },
        )?;
        let point = utils::execution_end_checkpoint(&outcome)?;
        device.wait_completion(&point).await?;
    }
    scope.finish().await;
    Ok(())
}

fn issue_frame(
    device: &Device<Gl>,
    lane: LaneId,
    workload: &mut Workload<Gl>,
    image: SurfaceImage<Gl>,
    extent: [u32; 2],
) -> Result<()> {
    let (_, outcome) = utils::execute_frame(
        device,
        lane,
        image,
        CompletionRequest::End,
        |encoder, image| {
            let ticket = workload.encode(device, encoder, image, extent)?;
            workload.retain_ticket(Some(ticket))?;
            Ok(())
        },
    )?;
    let point = utils::execution_end_checkpoint(&outcome)?;
    utils::block_on(device.wait_completion(&point))?;
    let ticket = workload
        .take_completed_ticket()
        .ok_or(Error::InvalidState)?;
    workload.consume(device, ticket, &point)
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
    utils::gl_api::run_native(
        "Occlusion queries",
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
                ))?);
            }
            let mut workload = workload.borrow_mut();
            let workload = workload.as_mut().ok_or(Error::InvalidState)?;
            issue_frame(device, lane, workload, image, extent)
        },
    )
}

#[cfg(not(any(windows, all(target_os = "linux", feature = "native-gles-egl"))))]
pub(super) fn run() -> Result<()> {
    Err(Error::Unsupported("11 GL WGL host"))
}

const _: fn() = || {
    fn assert_workload<B: OcclusionWorkloadBackend>() {}
    assert_workload::<Gl>();
};
