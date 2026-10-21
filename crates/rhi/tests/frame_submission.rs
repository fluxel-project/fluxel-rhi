//! Fault injection for the example host's submission lifetime policy.
#![allow(dead_code, unused_imports)]

#[path = "../examples/utils/mod.rs"]
mod utils;

use fluxel_rhi::api::*;
use std::{
    future::{Ready, ready},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Clone, Copy)]
enum Mode {
    Accepted,
    AcceptedCommandsCoverage,
    AcceptedMalformed,
    AcceptedDuplicate,
    AcceptedWrongPlan,
    AcceptedWrongPresent,
    AcceptedWrongCoverage,
    IdleFailure,
    Rejected,
    Partial,
    Indeterminate,
}
struct State {
    mode: Mode,
    waits: AtomicUsize,
    retained: AtomicUsize,
}
struct Token(Arc<State>);
impl Token {
    fn new(state: &Arc<State>) -> Self {
        state.retained.fetch_add(1, Ordering::SeqCst);
        Self(state.clone())
    }
}
impl Clone for Token {
    fn clone(&self) -> Self {
        Self::new(&self.0)
    }
}
impl Drop for Token {
    fn drop(&mut self) {
        self.0.retained.fetch_sub(1, Ordering::SeqCst);
    }
}
struct Mock;
// SAFETY: the mock has a single current generation and never performs native work.
unsafe impl NativeImageLease for Token {
    fn try_handoff(&mut self, _: u64) -> Result<()> {
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
    type NativeEncoder = ();
    type NativeCompletion = Token;
    type NativeSurface = Arc<State>;
    type NativeSurfaceImage = Token;
    type NativeUncertainty = Token;
    type Compute = Unavailable;
    type Mesh = Unavailable;
    type Ray = Unavailable;
    type Extras = Unavailable;
}
impl CommandBackend for Mock {
    fn begin_encoder(_: &Arc<State>, _: LaneId) -> Result<()> {
        Ok(())
    }
    fn transition(_: &mut (), _: &[ResourceTransition<'_, Self>]) -> Result<()> {
        Ok(())
    }
    fn copy_buffer(_: &mut (), _: &(), _: u64, _: &(), _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn copy_texture(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn buffer_to_texture(
        _: &mut (),
        _: &(),
        _: TexelCopyBufferLayout,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn texture_to_buffer(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: &(),
        _: TexelCopyBufferLayout,
    ) -> Result<()> {
        Ok(())
    }
    fn write_buffer(_: &mut (), _: &(), _: u64, _: &[u8]) -> Result<()> {
        Ok(())
    }
    fn write_texture(
        _: &mut (),
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
        assert!(
            state.retained.load(Ordering::SeqCst) > 0,
            "outcome payload must survive until idle"
        );
        state.waits.fetch_add(1, Ordering::SeqCst);
        if matches!(state.mode, Mode::IdleFailure) {
            return ready(Err(Error::DeviceLost));
        }
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
    fn acquire<'a>(surface: &'a mut Arc<State>) -> Self::AcquireFuture<'a> {
        ready(Ok(Token::new(surface)))
    }
}
impl DeferredBackend for Mock {
    type NativeRecordedWork = ();
    fn finish_encoder(_: (), _: LaneId) -> Result<()> {
        Ok(())
    }
    // SAFETY: this fixture has no GPU work and returns the selected synthetic outcome.
    unsafe fn submit(state: &Arc<State>, mut packet: SubmitPacket<Self>) -> SubmitOutcome<Self> {
        let device = packet.plan.id().device;
        let first = packet.plan.step(0);
        let second = packet.plan.step(1);
        let lane = packet.plan.steps()[0].lane;
        match state.mode {
            Mode::Rejected => SubmitOutcome::Rejected {
                error: Error::InvalidState,
                work: RetryableSubmit::Fresh(packet),
            },
            Mode::AcceptedMalformed => SubmitOutcome::Accepted(SubmissionReceipt {
                device,
                index: None,
                accepted: vec![first],
                presents: vec![],
                checkpoints: vec![Checkpoint {
                    // SAFETY: synthetic completion represents this mock device.
                    point: unsafe { CompletionPoint::from_native(device, Token::new(state)) },
                    covers: CompletionCoverage {
                        lane_frontiers: vec![],
                    },
                }],
            }),
            Mode::Accepted
            | Mode::AcceptedCommandsCoverage
            | Mode::AcceptedDuplicate
            | Mode::AcceptedWrongPlan
            | Mode::AcceptedWrongPresent
            | Mode::AcceptedWrongCoverage
            | Mode::IdleFailure => {
                let foreign = StepId {
                    plan: PlanId {
                        device,
                        ordinal: packet.plan.id().ordinal.wrapping_add(1),
                    },
                    index: 1,
                };
                let accepted = match state.mode {
                    Mode::AcceptedDuplicate => vec![first, first],
                    Mode::AcceptedWrongPlan => vec![first, foreign],
                    _ => vec![first, second],
                };
                let present = if matches!(state.mode, Mode::AcceptedWrongPresent) {
                    first
                } else {
                    second
                };
                let coverage = if matches!(state.mode, Mode::AcceptedCommandsCoverage) {
                    first
                } else if matches!(state.mode, Mode::AcceptedWrongCoverage) {
                    foreign
                } else {
                    second
                };
                SubmitOutcome::Accepted(SubmissionReceipt {
                    device,
                    index: None,
                    accepted,
                    presents: vec![PresentReceipt {
                        step: Some(present),
                        accepted: true,
                    }],
                    checkpoints: vec![Checkpoint {
                        // SAFETY: synthetic completion represents this mock device.
                        point: unsafe { CompletionPoint::from_native(device, Token::new(state)) },
                        covers: CompletionCoverage {
                            lane_frontiers: vec![(lane, coverage)],
                        },
                    }],
                })
            }
            Mode::Partial => {
                packet.steps.remove(0);
                let accepted = SubmissionReceipt {
                    device,
                    index: None,
                    accepted: vec![first],
                    checkpoints: vec![],
                    presents: vec![],
                };
                let checkpoint = Checkpoint {
                    // SAFETY: the mock completion covers its accepted command step.
                    point: unsafe { CompletionPoint::from_native(device, Token::new(state)) },
                    covers: CompletionCoverage {
                        lane_frontiers: vec![(packet.plan.steps()[0].lane, first)],
                    },
                };
                let tail = DeferredTail {
                    plan: packet.plan,
                    next_step: 1,
                    pending: packet.steps,
                    accepted_frontier: vec![checkpoint],
                    completion: packet.completion,
                };
                SubmitOutcome::PartiallyAccepted {
                    error: Error::InvalidState,
                    accepted,
                    tail,
                }
            }
            Mode::Indeterminate => SubmitOutcome::Indeterminate {
                error: Error::DeviceLost,
                // SAFETY: synthetic uncertainty belongs to this mock device.
                uncertain: unsafe { NativeUncertainty::from_native(device, Token::new(state)) },
            },
        }
    }
    unsafe fn resume(_: &Arc<State>, tail: DeferredTail<Self>) -> SubmitOutcome<Self> {
        SubmitOutcome::Rejected {
            error: Error::InvalidState,
            work: RetryableSubmit::Tail(tail),
        }
    }
}

impl MultiQueueBackend for Mock {
    type NativeGpuSignal = Token;

    // SAFETY: this fixture performs no GPU work. It models backend-only
    // consumption after Core validates the incoming packet.
    unsafe fn submit_multi(
        state: &Arc<State>,
        mut packet: MultiQueuePacket<Self>,
    ) -> MultiQueueSubmitOutcome<Self> {
        let wait = packet.waits.pop().expect("one mock wait");
        // SAFETY: the mock is the signal's owning backend; this transfers the
        // validated one-shot signal to the synthetic wait and never reuses it.
        let native = unsafe { wait.signal.into_native() };
        assert!(Arc::ptr_eq(&native.0, state));
        drop(native);
        MultiQueueSubmitOutcome::Accepted(MultiQueueReceipt {
            work: SubmissionReceipt {
                device: packet.work.plan.id().device,
                index: None,
                accepted: packet.work.steps.iter().map(|step| step.step).collect(),
                checkpoints: vec![],
                presents: vec![],
            },
            signals: vec![],
        })
    }
}

#[test]
fn external_backend_can_consume_validated_native_signal() {
    let state = Arc::new(State {
        mode: Mode::Accepted,
        waits: AtomicUsize::new(0),
        retained: AtomicUsize::new(0),
    });
    let id = DeviceId::allocate().unwrap();
    let producer_lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let consumer_lane = LaneId {
        device: id,
        ordinal: 1,
    };
    let caps = DeviceCapabilities {
        multi_queue: true,
        lanes: Arc::from([
            LaneInfo {
                id: producer_lane,
                class: LaneClass::Graphics,
                routes: vec![(consumer_lane, WaitRoute::NativeGpuWait)],
            },
            LaneInfo {
                id: consumer_lane,
                class: LaneClass::Graphics,
                routes: vec![],
            },
        ]),
        ..Default::default()
    };
    // SAFETY: this mock device, identity and capabilities are one fixture.
    let device = unsafe { Device::<Mock>::from_native(id, state.clone(), caps) };
    // SAFETY: the mock has no native queue effects and the consumer lane is valid.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: id,
                ordinal: 42,
            },
            Arc::from([StepDesc {
                lane: consumer_lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let producer = StepId {
        plan: PlanId {
            device: id,
            ordinal: 41,
        },
        index: 0,
    };
    let consumer = plan.step(0);
    // SAFETY: this test models a mock signal from the declared prior execution;
    // no real native signal or queue work exists in this fixture.
    let signal = unsafe {
        GpuSignalPoint::<Mock>::from_native(
            id,
            producer,
            producer_lane,
            consumer,
            consumer_lane,
            Token::new(&state),
        )
    };
    let work = device.encoder(consumer_lane).unwrap().finish().unwrap();
    let packet = MultiQueuePacket {
        work: SubmitPacket {
            steps: vec![BoundStep {
                step: consumer,
                payload: StepPayload::Commands(vec![work]),
            }],
            plan,
            completion: CompletionRequest::None,
        },
        signals: vec![],
        waits: vec![GpuWait {
            signal,
            destination: AccessScope {
                stages: StageMask::ALL_COMMANDS,
                accesses: AccessMask::ALL,
            },
        }],
    };
    assert_eq!(state.retained.load(Ordering::SeqCst), 1);
    // SAFETY: fixture signals and the selected consumer refer to the same mock execution.
    let outcome = unsafe { device.multi_queue().unwrap().submit(packet) };
    assert!(matches!(outcome, MultiQueueSubmitOutcome::Accepted(_)));
    assert_eq!(state.retained.load(Ordering::SeqCst), 0);
}

fn issue(mode: Mode) -> (Arc<State>, utils::FrameSubmission<Mock>) {
    let state = Arc::new(State {
        mode,
        waits: AtomicUsize::new(0),
        retained: AtomicUsize::new(0),
    });
    let id = DeviceId::allocate().unwrap();
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    // SAFETY: this isolated fixture associates its unique id with the mock.
    let device =
        unsafe { Device::<Mock>::from_native(id, state.clone(), DeviceCapabilities::default()) };
    // SAFETY: the mock surface belongs to the same fixture device.
    let mut surface =
        unsafe { Surface::<Mock>::from_native(id, SurfaceId::allocate().unwrap(), state.clone()) };
    let image = utils::block_on(surface.acquire()).unwrap();
    let work = device.encoder(lane).unwrap().finish().unwrap();
    let frame = utils::submit_frame(&device, lane, work, image).unwrap();
    (state, frame)
}

#[test]
fn clean_rejection_does_not_wait_idle() {
    let (state, frame) = issue(Mode::Rejected);
    assert!(matches!(frame.outcome(), SubmitOutcome::Rejected { .. }));
    let complete = utils::block_on(frame.complete());
    assert_eq!(state.waits.load(Ordering::SeqCst), 0);
    assert!(matches!(complete.outcome, SubmitOutcome::Rejected { .. }));
}

#[test]
fn accepted_malformed_partial_and_uncertain_drain_before_policy() {
    for mode in [Mode::AcceptedMalformed, Mode::Partial, Mode::Indeterminate] {
        let (state, frame) = issue(mode);
        assert_eq!(state.waits.load(Ordering::SeqCst), 0);
        let complete = utils::block_on(frame.complete());
        assert_eq!(state.waits.load(Ordering::SeqCst), 1);
        assert!(state.retained.load(Ordering::SeqCst) > 0);
        assert!(match (&complete.outcome, mode) {
            (SubmitOutcome::Accepted(_), Mode::AcceptedMalformed)
            | (SubmitOutcome::PartiallyAccepted { .. }, Mode::Partial)
            | (SubmitOutcome::Indeterminate { .. }, Mode::Indeterminate) => true,
            _ => false,
        });
        assert!(complete.into_presented().is_err());
        assert_eq!(state.retained.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn success_requires_a_receipt_for_the_submitted_frame() {
    for mode in [Mode::Accepted, Mode::AcceptedCommandsCoverage] {
        let (state, frame) = issue(mode);
        let receipt = utils::block_on(frame.complete()).into_presented().unwrap();
        assert_eq!(state.waits.load(Ordering::SeqCst), 1);
        assert_eq!(receipt.accepted.len(), 2);
        assert_eq!(receipt.presents[0].step, Some(receipt.accepted[1]));
        drop(receipt);
        assert_eq!(state.retained.load(Ordering::SeqCst), 0);
    }
    for mode in [
        Mode::AcceptedDuplicate,
        Mode::AcceptedWrongPlan,
        Mode::AcceptedWrongPresent,
        Mode::AcceptedWrongCoverage,
    ] {
        let (state, frame) = issue(mode);
        assert!(matches!(
            utils::block_on(frame.complete()).into_presented(),
            Err(Error::InvalidState)
        ));
        assert_eq!(state.waits.load(Ordering::SeqCst), 1);
        assert_eq!(state.retained.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn fail_stop_child() {
    let Ok(scenario) = std::env::var("FLUXEL_FAIL_STOP_SCENARIO") else {
        return;
    };
    struct SceneDrop;
    impl Drop for SceneDrop {
        fn drop(&mut self) {
            eprintln!("SCENE_DROPPED");
        }
    }
    let _scene = SceneDrop;
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        unsafe extern "system" {
            fn SetErrorMode(mode: u32) -> u32;
        }
        // SAFETY: these flags only suppress OS error-reporting dialogs in this child process.
        unsafe {
            SetErrorMode(0x0001 | 0x0002 | 0x8000);
        }
    }
    match scenario.as_str() {
        "cancel" => {
            let (_, frame) = issue(Mode::Accepted);
            drop(frame);
        }
        "idle_failure" => {
            let (_, frame) = issue(Mode::IdleFailure);
            let _ = utils::block_on(frame.complete());
        }
        _ => panic!("unknown test scenario"),
    }
    panic!("fail-stop returned normally");
}

#[test]
fn cancellation_and_failed_drain_terminate_without_unwinding() {
    for (scenario, expected) in [
        (
            "cancel",
            "submission observation cancelled before GPU drain",
        ),
        ("idle_failure", "frame drain failed"),
    ] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("fail_stop_child")
            .arg("--nocapture")
            .env("FLUXEL_FAIL_STOP_SCENARIO", scenario);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            command.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }
        let output = command.output().unwrap();
        assert!(
            !output.status.success(),
            "{scenario} must terminate the child"
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(expected),
            "unexpected child stderr: {stderr}"
        );
        assert!(!stderr.contains("fail-stop returned normally"));
        assert!(
            !stderr.contains("SCENE_DROPPED"),
            "scene unwound before termination: {stderr}"
        );
    }
}
