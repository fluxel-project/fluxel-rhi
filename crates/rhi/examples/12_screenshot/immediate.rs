//! Immediate GL paths for the shared screenshot workload.
//!
//! Texture capture uses the GL backend's direct readback route at the command
//! call site. The immediate session then supplies Core's completion evidence
//! before the ticket is exposed as a file.

use std::{cell::RefCell, rc::Rc};

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
use fluxel_rhi::gl::GlInstance;
use fluxel_rhi::{api::*, gl::Gl};

use super::{Screenshot, ScreenshotWorkloadBackend, utils};

fn issue_frame(
    device: &Device<Gl>,
    lane: LaneId,
    screenshot: &mut Screenshot<Gl>,
    image: SurfaceImage<Gl>,
    capture: bool,
) -> Result<CompletionPoint<Gl>> {
    screenshot.ensure_ticket_available()?;
    let (_, outcome) = utils::execute_frame(
        device,
        lane,
        image,
        CompletionRequest::End,
        |encoder, image| {
            let ticket = screenshot.encode(encoder, image, capture)?;
            screenshot.retain_ticket(ticket)?;
            Ok(())
        },
    )?;
    let point = utils::execution_end_checkpoint(&outcome)?;
    screenshot.commit_frame(capture);
    Ok(point)
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
    let capture_requested = std::env::var_os("FLUXEL_RHI_SCREENSHOT").is_some();
    let screenshot = Rc::new(RefCell::new(None::<(Screenshot<Gl>, [u32; 2])>));
    let dropped = Rc::clone(&screenshot);
    let captured = Rc::new(RefCell::new(false));
    let captured_frame = Rc::clone(&captured);
    utils::gl_api::run_native(
        "Screenshot readback",
        config.clone(),
        move || *dropped.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            if screenshot
                .borrow()
                .as_ref()
                .is_none_or(|(_, previous_extent)| *previous_extent != extent)
            {
                *screenshot.borrow_mut() = Some((
                    utils::block_on(Screenshot::<Gl>::new_with_surface_format(
                        device,
                        lane,
                        extent,
                        TextureFormat::Rgba8Unorm,
                    ))?,
                    extent,
                ));
            }
            let mut slot = screenshot.borrow_mut();
            let (workload, _) = slot.as_mut().ok_or(Error::InvalidState)?;
            let capture = capture_requested && !*captured_frame.borrow();
            let point = issue_frame(device, lane, workload, image, capture)?;
            utils::block_on(device.wait_completion(&point))?;
            if let Some(ticket) = workload.take_completed_ticket() {
                utils::block_on(workload.save_ppm_after(device, ticket, &point, "screenshot.ppm"))?;
                *captured_frame.borrow_mut() = true;
            }
            Ok(())
        },
    )
}

#[cfg(not(any(windows, all(target_os = "linux", feature = "native-gles-egl"))))]
pub(super) fn run() -> Result<()> {
    Err(Error::Unsupported("12 GL WGL host"))
}

/// Renders the shared workload and proves the direct WebGL2 texture-to-buffer
/// and buffer-readback route. Browser GL calls execute inside `issue_frame`;
/// awaiting the returned completion point only satisfies Core's ticket
/// visibility contract before host code inspects the copied bytes.
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
    // SAFETY: the canvas-backed owner stays retained by the instance, device,
    // and surface while each immediate command is issued on this browser thread.
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
    let mut surface = Gl::surface(&device)?;
    surface
        .configure(&SurfaceConfig {
            extent,
            format: TextureFormat::Rgba8Unorm,
            present_mode: PresentMode::Fifo,
            alpha_mode: CompositeAlphaMode::Opaque,
            color_space: ColorSpace::Srgb,
            usage: SurfaceUsage::RENDER_ATTACHMENT,
            maximum_frame_latency: 1,
        })
        .await?;
    let mut screenshot =
        Screenshot::<Gl>::new_with_surface_format(&device, lane, extent, TextureFormat::Rgba8Unorm)
            .await?;
    let scope = utils::GpuScope::new(&device);
    for frame in 0..frames {
        let image = surface.acquire().await?;
        let point = issue_frame(&device, lane, &mut screenshot, image, frame == 0)?;
        device.wait_completion(&point).await?;
        if let Some(ticket) = screenshot.take_completed_ticket() {
            // SAFETY: `point` is the End checkpoint returned by this exact
            // immediate session and was awaited immediately above.
            let data = unsafe { device.read_after(ticket, &point).await? };
            let ppm = super::ppm_rgb(&data.bytes, screenshot.row_pitch as usize, extent, false)?;
            if !ppm.starts_with(b"P6\n") || ppm.len() <= 64 {
                return Err(Error::InvalidState);
            }
            let row_bytes = extent[0].checked_mul(4).ok_or(Error::InvalidRange)? as usize;
            let first = data.bytes.get(..4).ok_or(Error::InvalidState)?;
            let changed = data
                .bytes
                .chunks_exact(screenshot.row_pitch as usize)
                .take(extent[1] as usize)
                .flat_map(|row| row[..row_bytes].chunks_exact(4))
                .filter(|pixel| *pixel != first)
                .count();
            if changed <= 64 {
                return Err(Error::InvalidState);
            }
        }
    }
    scope.finish().await;
    Ok(())
}

const _: fn() = || {
    fn assert_workload<B: ScreenshotWorkloadBackend>() {}
    assert_workload::<Gl>();
};
