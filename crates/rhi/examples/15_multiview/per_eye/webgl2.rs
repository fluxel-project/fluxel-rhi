//! WebGL2 immediate surface loop for the sequential stereo fallback.

use super::*;

/// Runs the ordinary two-pass stereo fallback on WebGL2's immediate context.
/// Native multiview remains unavailable and is not emulated through command replay.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub async fn run_webgl2_multiview(canvas: web_sys::HtmlCanvasElement, frames: u32) -> Result<()> {
    use fluxel_rhi::gl::{Gl, GlInstance};
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };

    static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);
    if frames == 0 {
        return Err(Error::InvalidRange);
    }
    let native = GlInstance::from_webgl2_canvas(canvas.clone())?;
    // SAFETY: the canvas-backed owner is retained by the typed instance,
    // device, and surface while every GL operation stays on this browser thread.
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
    if device.capabilities().multiview {
        return Err(Error::InvalidState);
    }
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
    let mut scene = StereoFallback::<Gl>::new(&device, format, extent).await?;
    for _ in 0..frames {
        let image = surface.acquire().await?;
        let ordinal = NEXT_PLAN
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| Error::OutOfMemory)?;
        // SAFETY: one command step and one present step execute on the same
        // browser graphics lane, in the precompiled order.
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
        // SAFETY: the encoder and presentation call run on the canvas owner
        // thread and finish before this iteration releases the acquired image.
        let mut session = unsafe { device.immediate(plan, CompletionRequest::None)? };
        let mut commands = session.begin_commands().map_err(|outcome| match outcome {
            ImmediateRunOutcome::Issued(_) => Error::InvalidState,
            ImmediateRunOutcome::Failed { error, .. }
            | ImmediateRunOutcome::Indeterminate { error, .. } => error,
        })?;
        let mut encoder = commands.commands();
        let uploaded = scene.encode(&mut encoder, &image)?;
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
        match session.finish() {
            ImmediateFinishOutcome::Issued(_) => {}
            ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Failed { error, .. })
            | ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Indeterminate {
                error, ..
            }) => {
                return Err(error);
            }
            ImmediateFinishOutcome::Failed(ImmediateRunOutcome::Issued(_))
            | ImmediateFinishOutcome::AlreadyReported => return Err(Error::InvalidState),
        }
    }
    Ok(())
}
