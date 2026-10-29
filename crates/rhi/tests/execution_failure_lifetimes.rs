//! Lifetime ordering of the shared example execution helper under failures.
#![allow(dead_code, unused_imports)]
#[path = "../examples/utils/mod.rs"]
mod utils;
use fluxel_rhi::api::*;
use fluxel_rhi::backend::*;
use std::{
    future::{Ready, ready},
    sync::{Arc, Mutex},
};

#[derive(Clone, Copy)]
enum Mode {
    EncodeFailure,
    PresentRejected,
    PresentUnknown,
    Success,
}
struct State {
    mode: Mode,
    events: Mutex<Vec<&'static str>>,
    id: DeviceId,
}
impl State {
    fn event(&self, name: &'static str) {
        self.events.lock().unwrap().push(name);
    }
}
#[derive(Clone)]
struct Token {
    state: Arc<State>,
    kind: &'static str,
}
impl Drop for Token {
    fn drop(&mut self) {
        self.state.event(self.kind);
    }
}
struct Uncertain {
    state: Arc<State>,
    binding: Box<PresentBinding<Mock>>,
}
impl Drop for Uncertain {
    fn drop(&mut self) {
        self.state.event("uncertainty_drop");
    }
}
struct Mock;
// SAFETY: the fixture uses one generation and retains its lease until handoff.
unsafe impl NativeImageLease for Token {
    fn try_handoff(&mut self, _: u64) -> Result<()> {
        self.state.event("handoff");
        Ok(())
    }
    fn is_current(&self, generation: u64) -> bool {
        generation == 1
    }
}
impl Backend for Mock {
    type NativeDevice = Arc<State>;
    type NativeBuffer = ();
    type NativeTexture = ();
    type NativeTextureView = ();
    type NativeVideoFrame = ();
    type NativeSampler = ();
    type NativeBindGroupLayout = ();
    type NativePipelineLayout = ();
    type NativeBindGroup = ();
    type NativeShader = ();
    type NativeRasterPipeline = ();
    type NativeComputePipeline = ();
    type NativeMeshPipeline = ();
    type NativeRayPipeline = ();
    type NativePipelineCache = ();
    type NativeAcceleration = ();
    type NativeSparseAllocation = ();
    type NativeQuerySet = ();
    type NativeEncoder = Arc<State>;
    type NativeCompletion = Token;
    type NativeSurface = Arc<State>;
    type NativeSurfaceImage = Token;
    type NativeUncertainty = Uncertain;
    type MemoryObservation = Unavailable;
    type Compute = Unavailable;
    type Mesh = Unavailable;
    type Ray = Unavailable;
    type Extras = Unavailable;
}
impl CommandBackend for Mock {
    fn begin_encoder(state: &Arc<State>, _: LaneId) -> Result<Arc<State>> {
        Ok(state.clone())
    }
    fn transition(state: &mut Arc<State>, _: &[ResourceTransition<'_, Self>]) -> Result<()> {
        state.event("native_command");
        Ok(())
    }
    fn copy_buffer(_: &mut Arc<State>, _: &(), _: u64, _: &(), _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn copy_texture(
        _: &mut Arc<State>,
        _: TextureRegion<'_, Self>,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn buffer_to_texture(
        _: &mut Arc<State>,
        _: &(),
        _: TexelCopyBufferLayout,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn texture_to_buffer(
        _: &mut Arc<State>,
        _: TextureRegion<'_, Self>,
        _: &(),
        _: TexelCopyBufferLayout,
    ) -> Result<()> {
        Ok(())
    }
    fn write_buffer(_: &mut Arc<State>, _: &(), _: u64, _: &[u8]) -> Result<()> {
        Ok(())
    }
    fn write_texture(
        _: &mut Arc<State>,
        _: TextureRegion<'_, Self>,
        _: TexelCopyBufferLayout,
        _: &[u8],
    ) -> Result<()> {
        Ok(())
    }
}

impl CompletionBackend for Mock {
    type WaitFuture<'a> = Ready<Result<()>>;
    fn completion_status(_: &Arc<State>, _: &Token) -> CompletionState {
        CompletionState::Complete
    }
    fn wait_completion<'a>(_: &'a Arc<State>, _: &'a Token) -> Self::WaitFuture<'a> {
        ready(Ok(()))
    }
    fn wait_idle<'a>(state: &'a Arc<State>) -> Self::WaitFuture<'a> {
        state.event("idle");
        ready(Ok(()))
    }
}
impl SurfaceBackend for Mock {
    type ConfigureFuture<'a> = Ready<Result<()>>;
    type AcquireFuture<'a> = Ready<Result<Token>>;
    fn generation(_: &Arc<State>) -> u64 {
        1
    }
    fn capabilities(_: &Arc<State>) -> Result<SurfaceCapabilities> {
        Err(Error::Unsupported("mock"))
    }
    fn supports_config(_: &Arc<State>, _: &SurfaceConfig) -> Support<()> {
        Support::Unsupported("mock")
    }
    fn configure<'a>(_: &'a mut Arc<State>, _: &'a SurfaceConfig) -> Self::ConfigureFuture<'a> {
        ready(Ok(()))
    }
    fn acquire<'a>(state: &'a mut Arc<State>) -> Self::AcquireFuture<'a> {
        ready(Ok(Token {
            state: state.clone(),
            kind: "image_drop",
        }))
    }
}
impl ExecutionBackend for Mock {
    type Mode = ImmediateExecution;
}
// SAFETY: the mock records exactly which native operations have issued.
unsafe impl ImmediateBackend for Mock {
    fn install_waits(_: &Arc<State>, _: LaneId, _: &[StepWait]) -> WaitInstallOutcome<Self> {
        WaitInstallOutcome::Ready { issued: false }
    }
    fn close_immediate(state: Arc<State>, reason: ImmediateCloseReason) -> ImmediateClose<Self> {
        state.event("close");
        match reason {
            ImmediateCloseReason::Finished => ImmediateClose::Issued,
            ImmediateCloseReason::Aborted => ImmediateClose::Failed {
                error: Error::InvalidState,
                issued: true,
            },
        }
    }
    fn present_immediate(
        state: &Arc<State>,
        _: LaneId,
        _: &[StepWait],
        mut binding: PresentBinding<Self>,
    ) -> std::result::Result<ImmediateClose<Self>, (Error, PresentBinding<Self>)> {
        state.event("present");
        match state.mode {
            Mode::PresentRejected => Err((Error::InvalidState, binding)),
            Mode::PresentUnknown => {
                // SAFETY: the mock models the one irrevocable native handoff.
                unsafe { binding.try_handoff().unwrap() };
                Ok(ImmediateClose::Indeterminate {
                    error: Error::DeviceLost,
                    uncertain: NativeUncertainty {
                        device: state.id,
                        native: Uncertain {
                            state: state.clone(),
                            binding: Box::new(binding),
                        },
                    },
                })
            }
            _ => {
                drop(binding);
                Ok(ImmediateClose::Issued)
            }
        }
    }
    fn checkpoints(
        _: &Arc<State>,
        _: &SubmissionPlan<Self>,
        _: &[StepId],
        _: &CompletionRequest,
    ) -> CheckpointOutcome<Self> {
        CheckpointOutcome::Covered(vec![])
    }
    fn emergency_checkpoint(
        state: &Arc<State>,
        plan: &SubmissionPlan<Self>,
        issued: &[StepId],
    ) -> CheckpointOutcome<Self> {
        let Some(step) = issued.last().copied() else {
            return CheckpointOutcome::Covered(vec![]);
        };
        // SAFETY: the mock checkpoint covers the issued native prefix on this fixture device.
        let point = unsafe {
            CompletionPoint::from_native(
                state.id,
                Token {
                    state: state.clone(),
                    kind: "checkpoint_drop",
                },
            )
        };
        CheckpointOutcome::Covered(vec![Checkpoint {
            point,
            covers: CompletionCoverage {
                lane_frontiers: vec![(plan.steps()[step.index as usize].lane, step)],
            },
        }])
    }
    fn abandon_session(_: &Arc<State>, _: &SubmissionPlan<Self>, _: &[StepId]) {}
}
fn run(mode: Mode) -> Vec<&'static str> {
    let id = DeviceId::allocate().unwrap();
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let state = Arc::new(State {
        mode,
        events: Mutex::new(vec![]),
        id,
    });
    let caps = DeviceCapabilities {
        lanes: Arc::from([LaneInfo {
            id: lane,
            class: LaneClass::Graphics,
            routes: vec![],
        }]),
        ..Default::default()
    };
    // SAFETY: native objects and identifiers belong to this one fixture.
    let device = unsafe { Device::<Mock>::from_native(id, state.clone(), caps) };
    // SAFETY: the surface is owned by the same synthetic device.
    let mut surface = unsafe {
        Surface::<Mock>::from_native(&device, SurfaceId::allocate().unwrap(), state.clone())
    };
    let image = utils::block_on(surface.acquire()).unwrap();
    // The caller owns scene resources, including on encode failure.
    let owner = Token {
        state: state.clone(),
        kind: "owner_drop",
    };
    let result = utils::execute_frame(
        &device,
        lane,
        image,
        CompletionRequest::None,
        |encoder, _| {
            encoder.transition_resources(&[])?;
            if matches!(mode, Mode::EncodeFailure) {
                return Err(Error::InvalidState);
            }
            Ok(Token {
                state: state.clone(),
                kind: "value_drop",
            })
        },
    );
    assert_eq!(result.is_ok(), matches!(mode, Mode::Success));
    drop(result);
    drop(owner);
    drop(surface);
    drop(device);
    let events = state.events.lock().unwrap().clone();
    events
}
fn before(events: &[&str], first: &str, second: &str) {
    let a = events.iter().position(|e| *e == first).expect(first);
    let b = events.iter().position(|e| *e == second).expect(second);
    assert!(a < b, "{events:?}");
}
#[test]
fn encode_failure_retains_owner_and_image_until_idle() {
    let e = run(Mode::EncodeFailure);
    before(&e, "native_command", "idle");
    before(&e, "idle", "image_drop");
    before(&e, "idle", "checkpoint_drop");
    before(&e, "idle", "owner_drop");
}
#[test]
fn present_rejection_retains_owner_and_lease_until_idle() {
    let e = run(Mode::PresentRejected);
    before(&e, "native_command", "present");
    before(&e, "present", "idle");
    before(&e, "idle", "image_drop");
    before(&e, "idle", "value_drop");
    before(&e, "idle", "owner_drop");
}
#[test]
fn unknown_present_retains_uncertainty_until_idle() {
    let e = run(Mode::PresentUnknown);
    before(&e, "present", "handoff");
    assert_eq!(e.iter().filter(|event| **event == "handoff").count(), 1);
    before(&e, "handoff", "idle");
    before(&e, "idle", "value_drop");
    before(&e, "idle", "uncertainty_drop");
    before(&e, "idle", "image_drop");
    before(&e, "idle", "owner_drop");
}
#[test]
fn success_does_not_wait_idle() {
    let e = run(Mode::Success);
    assert!(!e.contains(&"idle"), "{e:?}");
}
