//! WGL immediate route for the fixed texture-array fallback.
//!
//! GL does not claim runtime-sized non-uniform descriptor indexing. It uses
//! the same 32-layer WGSL texture-array visual path as WebGPU and issues each
//! RHI operation directly on the current context owner thread.

use std::{
    cell::RefCell,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use fluxel_rhi::{api::*, gl::Gl};

use super::{DescriptorIndexing, utils};

static NEXT_PLAN: AtomicU64 = AtomicU64::new(1);

fn outcome_error(outcome: ImmediateRunOutcome<Gl>) -> Error {
    match outcome {
        ImmediateRunOutcome::Issued(_) => Error::InvalidState,
        ImmediateRunOutcome::Failed { error, .. }
        | ImmediateRunOutcome::Indeterminate { error, .. } => error,
    }
}

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
    let start = Instant::now();
    let scene = Rc::new(RefCell::new(None::<DescriptorIndexing<Gl>>));
    let dropped = Rc::clone(&scene);
    utils::gl_api::run_wgl(
        "Fixed texture-array fallback",
        config.clone(),
        move || *dropped.borrow_mut() = None,
        move |device, _, image, extent| {
            let lane = device
                .lanes()
                .iter()
                .find(|lane| lane.class == LaneClass::Graphics)
                .map(|lane| lane.id)
                .ok_or(Error::Unsupported("GL graphics lane"))?;
            if scene.borrow().is_none() {
                let workload = utils::block_on(DescriptorIndexing::<Gl>::new(
                    device,
                    lane,
                    config.format,
                    extent,
                    true,
                ))?;
                *scene.borrow_mut() = Some(workload);
            }
            let ordinal = NEXT_PLAN
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                    value.checked_add(1)
                })
                .map_err(|_| Error::OutOfMemory)?;
            // SAFETY: the WGL callback owns the current context and emits one
            // Commands step followed by its Present step on the same lane.
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
            // SAFETY: every command and present call below remains on this
            // callback's context-owning thread until the session is finished.
            let mut session = unsafe { device.immediate(plan, CompletionRequest::End)? };
            let mut commands = session.begin_commands().map_err(outcome_error)?;
            let uploaded = scene
                .borrow_mut()
                .as_mut()
                .ok_or(Error::InvalidState)?
                .encode(
                    device,
                    &image,
                    commands.commands(),
                    extent,
                    start.elapsed().as_secs_f32(),
                )?;
            match commands.close() {
                ImmediateRunOutcome::Issued(_) => {}
                outcome => return Err(outcome_error(outcome)),
            }
            if uploaded {
                scene
                    .borrow_mut()
                    .as_mut()
                    .ok_or(Error::InvalidState)?
                    .mark_uploaded();
            }
            match session.present(PresentBinding::new(image)) {
                ImmediatePresentResult::Run(ImmediateRunOutcome::Issued(_)) => {}
                ImmediatePresentResult::Run(outcome) => return Err(outcome_error(outcome)),
                ImmediatePresentResult::Rejected { error, .. } => return Err(error),
            }
            let point = utils::immediate_end_checkpoint(session.finish())?;
            utils::block_on(device.wait_completion(&point))
        },
    )
}
