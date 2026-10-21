//! Core execution contract tests. The witness observes SPI calls, not a GPU.
use super::*;
use std::{
    cell::RefCell,
    future::{Future, Ready, ready},
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Waker},
};

struct Witness;
impl Backend for Witness {
    type NativeDevice = Arc<()>;
    type NativeBuffer = Arc<()>;
    type NativeTexture = Arc<()>;
    type NativeTextureView = Arc<()>;
    type NativeSampler = Arc<()>;
    type NativeBindGroupLayout = Arc<()>;
    type NativePipelineLayout = Arc<()>;
    type NativeBindGroup = Arc<()>;
    type NativeShader = Arc<()>;
    type NativeRasterPipeline = Arc<()>;
    type NativeComputePipeline = Arc<()>;
    type NativeMeshPipeline = Arc<()>;
    type NativeRayPipeline = Arc<()>;
    type NativePipelineCache = Arc<()>;
    type NativeAcceleration = Arc<()>;
    type NativeSparseAllocation = Arc<()>;
    type NativeQuerySet = Arc<()>;
    type NativeEncoder = ();
    type NativeCompletion = Arc<()>;
    type NativeSurface = Arc<()>;
    type NativeSurfaceImage = ();
    type NativeUncertainty = Token;
    type Compute = Unavailable;
    type Mesh = Unavailable;
    type Ray = Unavailable;
    type Extras = Unavailable;
}
impl CommandBackend for Witness {
    fn begin_encoder(_: &Arc<()>, _: LaneId) -> Result<()> {
        Ok(())
    }
    fn transition(_: &mut (), _: &[ResourceTransition<'_, Self>]) -> Result<()> {
        Ok(())
    }
    fn copy_buffer(_: &mut (), _: &Arc<()>, _: u64, _: &Arc<()>, _: u64, _: u64) -> Result<()> {
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
        _: &Arc<()>,
        _: TexelCopyBufferLayout,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn texture_to_buffer(
        _: &mut (),
        _: TextureRegion<'_, Self>,
        _: &Arc<()>,
        _: TexelCopyBufferLayout,
    ) -> Result<()> {
        Ok(())
    }
    fn write_buffer(_: &mut (), _: &Arc<()>, _: u64, _: &[u8]) -> Result<()> {
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

type SubmissionTrace = (Vec<(u32, Vec<u32>)>, Vec<u32>, &'static str);

#[derive(Default)]
struct Trace {
    calls: Vec<SubmissionTrace>,
    mode: u8,
    dropped: usize,
    waited: usize,
}
thread_local! { static TRACE: RefCell<Trace> = RefCell::new(Trace::default()); }
struct Token;
impl Drop for Token {
    fn drop(&mut self) {
        TRACE.with_borrow_mut(|t| t.dropped += 1);
    }
}
impl CompletionBackend for Witness {
    type WaitFuture<'a> = Ready<Result<()>>;
    fn completion_status(_: &Arc<()>, _: &Arc<()>) -> CompletionState {
        CompletionState::Complete
    }
    fn wait_completion<'a>(_: &'a Arc<()>, _: &'a Arc<()>) -> Self::WaitFuture<'a> {
        TRACE.with_borrow_mut(|t| t.waited += 1);
        ready(Ok(()))
    }
    fn wait_idle<'a>(_: &'a Arc<()>) -> Self::WaitFuture<'a> {
        ready(Ok(()))
    }
}
fn snapshot(
    plan: &SubmissionPlan<Witness>,
    steps: &[BoundStep<Witness>],
    request: &CompletionRequest,
) {
    let topology = plan
        .steps()
        .iter()
        .enumerate()
        .map(|(i, s)| (i as u32, s.waits.iter().map(|w| w.previous.index).collect()))
        .collect();
    let payload = steps
        .iter()
        .flat_map(|s| match &s.payload {
            StepPayload::Commands(w) => w.iter().map(|w| w.native).collect::<Vec<_>>(),
            _ => unreachable!(),
        })
        .collect();
    let request = match request {
        CompletionRequest::None => "none",
        CompletionRequest::End => "end",
        CompletionRequest::Steps(_) => "steps",
    };
    TRACE.with_borrow_mut(|t| t.calls.push((topology, payload, request)));
}
fn receipt(device: DeviceId, accepted: Vec<StepId>) -> SubmissionReceipt<Witness> {
    SubmissionReceipt {
        device,
        index: None,
        accepted,
        checkpoints: vec![],
        presents: vec![],
    }
}
impl DeferredBackend for Witness {
    type NativeRecordedWork = u32;
    fn finish_encoder(_: (), _: LaneId) -> Result<u32> {
        Ok(0)
    }
    unsafe fn submit(_: &Arc<()>, mut packet: SubmitPacket<Self>) -> SubmitOutcome<Self> {
        snapshot(&packet.plan, &packet.steps, &packet.completion);
        let device = packet.plan.id().device;
        let mode = TRACE.with_borrow(|t| t.mode);
        if mode == 1 {
            let first = packet.steps.remove(0);
            let frontier = Checkpoint {
                point: unsafe { CompletionPoint::from_native(device, Arc::new(())) },
                covers: CompletionCoverage {
                    lane_frontiers: vec![(packet.plan.steps()[0].lane, first.step)],
                },
            };
            return SubmitOutcome::PartiallyAccepted {
                error: Error::Backend("partial".into()),
                accepted: receipt(device, vec![first.step]),
                tail: DeferredTail {
                    plan: packet.plan,
                    next_step: 1,
                    pending: packet.steps,
                    accepted_frontier: vec![frontier],
                    completion: packet.completion,
                },
            };
        }
        if mode == 2 {
            return SubmitOutcome::Indeterminate {
                error: Error::Backend("unknown".into()),
                uncertain: unsafe { NativeUncertainty::from_native(device, Token) },
            };
        }
        SubmitOutcome::Accepted(receipt(
            device,
            packet.steps.iter().map(|s| s.step).collect(),
        ))
    }
    unsafe fn resume(_: &Arc<()>, tail: DeferredTail<Self>) -> SubmitOutcome<Self> {
        snapshot(&tail.plan, &tail.pending, &tail.completion);
        SubmitOutcome::Accepted(receipt(
            tail.plan.id().device,
            tail.pending.iter().map(|s| s.step).collect(),
        ))
    }
}
fn fixture() -> (Device<Witness>, LaneId, SubmissionPlan<Witness>) {
    TRACE.with_borrow_mut(|t| *t = Trace::default());
    let id = DeviceId::allocate().unwrap();
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let plan_id = PlanId {
        device: id,
        ordinal: 42,
    };
    let scope = AccessScope {
        stages: StageMask::ALL_COMMANDS,
        accesses: AccessMask::ALL,
    };
    let steps = vec![
        StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![],
        },
        StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![StepWait {
                previous: StepId {
                    plan: plan_id,
                    index: 0,
                },
                destination: scope,
            }],
        },
        StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![
                StepWait {
                    previous: StepId {
                        plan: plan_id,
                        index: 0,
                    },
                    destination: scope,
                },
                StepWait {
                    previous: StepId {
                        plan: plan_id,
                        index: 1,
                    },
                    destination: scope,
                },
            ],
        },
    ];
    // SAFETY: this witness plan is a single-lane ordered compiled sequence.
    let plan = unsafe { SubmissionPlan::from_compiled(plan_id, steps.into()) };
    // SAFETY: fake native device is paired with the freshly allocated ID.
    let device = unsafe { Device::from_native(id, Arc::new(()), DeviceCapabilities::default()) };
    (device, lane, plan)
}
fn packet(
    plan: &SubmissionPlan<Witness>,
    lane: LaneId,
    base: u32,
    completion: CompletionRequest,
) -> SubmitPacket<Witness> {
    let steps = (0..3)
        .map(|i| BoundStep {
            step: plan.step(i),
            payload: StepPayload::Commands(vec![RecordedWork {
                device: lane.device,
                lane,
                native: base + i as u32,
            }]),
        })
        .collect();
    SubmitPacket {
        plan: plan.clone(),
        steps,
        completion,
    }
}
fn wait(f: impl Future<Output = Result<()>>) {
    let mut f = pin!(f);
    assert!(matches!(
        f.as_mut().poll(&mut Context::from_waker(Waker::noop())),
        Poll::Ready(Ok(()))
    ));
}
#[test]
fn compiled_order_and_waits_reach_spi_unchanged_for_each_execution() {
    let (device, lane, plan) = fixture();
    let first = packet(&plan, lane, 10, CompletionRequest::None);
    let second = packet(
        &plan,
        lane,
        20,
        CompletionRequest::Steps(Arc::from([plan.step(1)])),
    );
    // SAFETY: the fixture supplies no referenced GPU resources and its compiled waits are valid.
    assert!(matches!(
        unsafe { device.queue().submit(first) },
        SubmitOutcome::Accepted(_)
    ));
    assert!(matches!(
        unsafe { device.queue().submit(second) },
        SubmitOutcome::Accepted(_)
    ));
    TRACE.with_borrow(|t| {
        assert_eq!(
            t.calls,
            vec![
                (
                    vec![(0, vec![]), (1, vec![0]), (2, vec![0, 1])],
                    vec![10, 11, 12],
                    "none"
                ),
                (
                    vec![(0, vec![]), (1, vec![0]), (2, vec![0, 1])],
                    vec![20, 21, 22],
                    "steps"
                )
            ]
        );
    });
}
#[test]
fn invalid_inputs_never_reach_spi_and_tail_can_resume_after_frontier() {
    let (device, lane, plan) = fixture();
    let mut bad = packet(&plan, lane, 1, CompletionRequest::End);
    bad.steps.swap(0, 1);
    assert!(matches!(
        unsafe { device.queue().submit(bad) },
        SubmitOutcome::Rejected {
            error: Error::InvalidState,
            ..
        }
    ));
    let other = DeviceId::allocate().unwrap();
    let mut bad = packet(&plan, lane, 1, CompletionRequest::End);
    if let StepPayload::Commands(work) = &mut bad.steps[0].payload {
        work[0].device = other;
    }
    assert!(matches!(
        unsafe { device.queue().submit(bad) },
        SubmitOutcome::Rejected {
            error: Error::WrongDevice,
            ..
        }
    ));
    TRACE.with_borrow_mut(|t| {
        assert!(t.calls.is_empty());
        t.mode = 1;
    });
    let outcome = unsafe {
        device
            .queue()
            .submit(packet(&plan, lane, 10, CompletionRequest::None))
    };
    let mut tail = match outcome {
        SubmitOutcome::PartiallyAccepted { accepted, tail, .. } => {
            assert_eq!(accepted.accepted, vec![plan.step(0)]);
            tail
        }
        _ => panic!("expected partial"),
    };
    for checkpoint in &tail.accepted_frontier {
        wait(device.wait_completion(&checkpoint.point));
    }
    tail.pending.swap(0, 1);
    let outcome = unsafe { device.queue().resume(tail) };
    let mut tail = match outcome {
        SubmitOutcome::Rejected {
            error: Error::InvalidState,
            work: RetryableSubmit::Tail(tail),
        } => tail,
        _ => panic!("expected rejected malformed tail"),
    };
    tail.pending.swap(0, 1);
    // The checkpoint is supplied by the mock and already observed above. Core
    // must reject altered device provenance before calling the SPI.
    tail.accepted_frontier[0].point.device = other;
    let outcome = unsafe { device.queue().resume(tail) };
    let mut tail = match outcome {
        SubmitOutcome::Rejected {
            error: Error::WrongDevice,
            work: RetryableSubmit::Tail(tail),
        } => tail,
        _ => panic!("expected rejected foreign frontier"),
    };
    tail.accepted_frontier[0].point.device = device.id();
    tail.accepted_frontier[0].covers.lane_frontiers[0].1 = plan.step(1);
    let outcome = unsafe { device.queue().resume(tail) };
    let mut tail = match outcome {
        SubmitOutcome::Rejected {
            error: Error::InvalidState,
            work: RetryableSubmit::Tail(tail),
        } => tail,
        _ => panic!("expected rejected frontier outside accepted prefix"),
    };
    tail.accepted_frontier[0].covers.lane_frontiers[0].1 = plan.step(0);
    // SAFETY: the original checkpoint was observed and the fixture owns no GPU resources.
    assert!(matches!(
        unsafe { device.queue().resume(tail) },
        SubmitOutcome::Accepted(_)
    ));
    TRACE.with_borrow(|t| {
        assert_eq!(t.waited, 1);
        assert_eq!(t.calls.len(), 2);
        assert_eq!(t.calls[1].1, vec![11, 12]);
        assert_eq!(t.calls[0].0, t.calls[1].0);
    });
}
#[test]
fn indeterminate_token_stays_owned_until_caller_drains() {
    let (device, lane, plan) = fixture();
    TRACE.with_borrow_mut(|t| t.mode = 2);
    let outcome = unsafe {
        device
            .queue()
            .submit(packet(&plan, lane, 10, CompletionRequest::End))
    };
    let uncertain = match outcome {
        SubmitOutcome::Indeterminate { uncertain, .. } => uncertain,
        _ => panic!("expected uncertainty"),
    };
    TRACE.with_borrow(|t| assert_eq!(t.dropped, 0));
    // The witness idle future models caller observation, not real GPU completion.
    wait(device.wait_idle());
    drop(uncertain);
    TRACE.with_borrow(|t| assert_eq!(t.dropped, 1));
}

#[test]
fn supplied_multi_lane_frontiers_are_validated_without_scheduling() {
    let (device, lane_a, _) = fixture();
    let lane_b = LaneId {
        device: device.id(),
        ordinal: 1,
    };
    let id = PlanId {
        device: device.id(),
        ordinal: 43,
    };
    // The test supplies this order and its waits as an already compiled plan.
    // This witness validates Core routing only, not native multi-queue support.
    let plan = unsafe {
        SubmissionPlan::<Witness>::from_compiled(
            id,
            Arc::from([
                StepDesc {
                    lane: lane_a,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane: lane_b,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane: lane_a,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
            ]),
        )
    };
    let make_tail = || DeferredTail {
        plan: plan.clone(),
        next_step: 2,
        pending: vec![BoundStep {
            step: plan.step(2),
            payload: StepPayload::Commands(vec![RecordedWork {
                device: device.id(),
                lane: lane_a,
                native: 88,
            }]),
        }],
        accepted_frontier: vec![Checkpoint {
            // SAFETY: this is a local witness point for this fake device.
            point: unsafe { CompletionPoint::from_native(device.id(), Arc::new(())) },
            covers: CompletionCoverage {
                lane_frontiers: vec![(lane_a, plan.step(0)), (lane_b, plan.step(1))],
            },
        }],
        completion: CompletionRequest::None,
    };
    for variant in 0..3 {
        let mut tail = make_tail();
        wait(device.wait_completion(&tail.accepted_frontier[0].point));
        let frontier = &mut tail.accepted_frontier[0].covers.lane_frontiers[1];
        let expected = match variant {
            0 => {
                frontier.0 = lane_a;
                Error::WrongLane
            }
            1 => {
                frontier.1.plan.ordinal += 1;
                Error::InvalidState
            }
            _ => {
                frontier.1 = plan.step(2);
                Error::InvalidState
            }
        };
        // SAFETY: the synthetic checkpoint was observed; no GPU objects exist.
        let outcome = unsafe { device.queue().resume(tail) };
        match outcome {
            SubmitOutcome::Rejected {
                error,
                work: RetryableSubmit::Tail(_),
            } => {
                assert_eq!(format!("{error:?}"), format!("{expected:?}"));
            }
            _ => panic!("invalid frontier must be rejected before SPI"),
        }
        TRACE.with_borrow(|t| assert!(t.calls.is_empty()));
    }
    let tail = make_tail();
    wait(device.wait_completion(&tail.accepted_frontier[0].point));
    // SAFETY: the witness checkpoint was observed and carries no GPU resources.
    assert!(matches!(
        unsafe { device.queue().resume(tail) },
        SubmitOutcome::Accepted(_)
    ));
    TRACE.with_borrow(|t| {
        assert_eq!(t.calls.len(), 1);
        assert_eq!(t.calls[0].1, vec![88]);
    });
}
