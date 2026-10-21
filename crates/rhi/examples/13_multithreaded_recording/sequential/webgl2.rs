//! WebGL2 immediate execution for the sequential fallback.

use super::*;

/// Runs the sequential fallback through WebGL2's owner-thread immediate SPI.
/// No worker or secondary-command capability is advertised or emulated.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub async fn run_webgl2_multithreaded_recording(
    canvas: web_sys::HtmlCanvasElement,
    frames: u32,
) -> Result<()> {
    use fluxel_rhi::gl::{Gl, GlInstance};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);
    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    let native = GlInstance::from_webgl2_canvas(canvas.clone())?;
    // SAFETY: the browser canvas remains owned by this future; all GL access
    // stays on the JavaScript owner thread through the typed device/surface.
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
    let mut scene = Sequential::<Gl>::new(&device, format).await?;
    let scope = utils::GpuScope::new(&device);
    for _ in 0..frames {
        let image = surface.acquire().await?;
        let ordinal = NEXT_PLAN
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| Error::OutOfMemory)?;
        // SAFETY: this single-lane plan orders direct commands before the
        // matching surface present on the current browser owner thread.
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
        // SAFETY: the image and workload stay alive until immediate commands
        // close and the end checkpoint completes below.
        let mut session = unsafe { device.immediate(plan, CompletionRequest::End)? };
        let mut commands = session.begin_commands().map_err(|outcome| match outcome {
            ImmediateRunOutcome::Issued(_) => Error::InvalidState,
            ImmediateRunOutcome::Failed { error, .. }
            | ImmediateRunOutcome::Indeterminate { error, .. } => error,
        })?;
        let mut encoder = commands.commands();
        let uploaded = scene.encode(&device, &image, &mut encoder, extent)?;
        drop(encoder);
        match commands.close() {
            ImmediateRunOutcome::Issued(_) => {
                if uploaded {
                    scene.mark_uploaded();
                }
            }
            ImmediateRunOutcome::Failed { error, .. }
            | ImmediateRunOutcome::Indeterminate { error, .. } => return Err(error),
        }
        match session.present(PresentBinding::new(image)) {
            ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
            ImmediatePresentResult::Run(ImmediateRunOutcome::Failed { error, .. })
            | ImmediatePresentResult::Run(ImmediateRunOutcome::Indeterminate { error, .. })
            | ImmediatePresentResult::Rejected { error, .. } => return Err(error),
        }
        let point = utils::immediate_end_checkpoint(session.finish())?;
        device.wait_completion(&point).await?;
    }
    scope.finish().await;
    Ok(())
}
