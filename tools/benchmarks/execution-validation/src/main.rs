//! Synthetic CPU-only diagnostics for Core plan validation and immediate Steps.
//!
//! The immediate workload closes a command encoder for every plan step and
//! asks the mock backend for a distinct checkpoint at every requested step.
//! It therefore exercises `ImmediateSession` request membership, checkpoint
//! storage, and final receipt construction without claiming any GPU result.

use fluxel_rhi_core::api::*;
use std::{
    future::{Ready, ready},
    hint::black_box,
    rc::Rc,
    sync::Arc,
    time::Instant,
};

struct ValidationMock;
struct ImmediateMock;
struct Lease;
struct ImmediateLease;

// SAFETY: the validation diagnostic never constructs or uses a surface image.
unsafe impl NativeImageLease for Lease {
    fn try_handoff(&mut self, _: u64) -> Result<()> {
        unreachable!()
    }
    fn is_current(&self, _: u64) -> bool {
        unreachable!()
    }
}

// SAFETY: the immediate workload has no present steps and never constructs an
// image lease. This witness satisfies the Backend type contract only.
unsafe impl NativeImageLease for ImmediateLease {
    fn try_handoff(&mut self, _: u64) -> Result<()> {
        unreachable!()
    }
    fn is_current(&self, _: u64) -> bool {
        unreachable!()
    }
}

macro_rules! backend {
    ($name:ident, $native:ty, $encoder:ty, $image:ty) => {
        impl Backend for $name {
            type NativeDevice = $native;
            type NativeBuffer = $native;
            type NativeTexture = $native;
            type NativeTextureView = $native;
            type NativeSampler = $native;
            type NativeBindGroupLayout = $native;
            type NativePipelineLayout = $native;
            type NativeBindGroup = $native;
            type NativeShader = $native;
            type NativeRasterPipeline = $native;
            type NativeComputePipeline = $native;
            type NativeMeshPipeline = $native;
            type NativeRayPipeline = $native;
            type NativePipelineCache = $native;
            type NativeAcceleration = $native;
            type NativeSparseAllocation = $native;
            type NativeQuerySet = $native;
            type NativeEncoder = $encoder;
            type NativeCompletion = $native;
            type NativeSurface = $native;
            type NativeSurfaceImage = $image;
            type NativeUncertainty = ();
            type Compute = Unavailable;
            type Mesh = Unavailable;
            type Ray = Unavailable;
            type Extras = Unavailable;
        }
    };
}

backend!(ValidationMock, Arc<()>, (), Lease);
backend!(ImmediateMock, Rc<()>, Rc<()>, ImmediateLease);

impl CommandBackend for ImmediateMock {
    fn begin_encoder(_: &Rc<()>, _: LaneId) -> Result<Rc<()>> {
        Ok(Rc::new(()))
    }
    fn transition(_: &mut Rc<()>, _: &[ResourceTransition<'_, Self>]) -> Result<()> {
        Ok(())
    }
    fn copy_buffer(_: &mut Rc<()>, _: &Rc<()>, _: u64, _: &Rc<()>, _: u64, _: u64) -> Result<()> {
        Ok(())
    }
    fn copy_texture(
        _: &mut Rc<()>,
        _: TextureRegion<'_, Self>,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn buffer_to_texture(
        _: &mut Rc<()>,
        _: &Rc<()>,
        _: TexelCopyBufferLayout,
        _: TextureRegion<'_, Self>,
    ) -> Result<()> {
        Ok(())
    }
    fn texture_to_buffer(
        _: &mut Rc<()>,
        _: TextureRegion<'_, Self>,
        _: &Rc<()>,
        _: TexelCopyBufferLayout,
    ) -> Result<()> {
        Ok(())
    }
    fn write_buffer(_: &mut Rc<()>, _: &Rc<()>, _: u64, _: &[u8]) -> Result<()> {
        Ok(())
    }
    fn write_texture(
        _: &mut Rc<()>,
        _: TextureRegion<'_, Self>,
        _: TexelCopyBufferLayout,
        _: &[u8],
    ) -> Result<()> {
        Ok(())
    }
}

impl CompletionBackend for ImmediateMock {
    type WaitFuture<'a> = Ready<Result<()>>;
    fn completion_status(_: &Rc<()>, _: &Rc<()>) -> CompletionState {
        CompletionState::Complete
    }
    fn wait_completion<'a>(_: &'a Rc<()>, _: &'a Rc<()>) -> Self::WaitFuture<'a> {
        ready(Ok(()))
    }
    fn wait_idle<'a>(_: &'a Rc<()>) -> Self::WaitFuture<'a> {
        ready(Ok(()))
    }
}

// SAFETY: the mock completes every close synchronously and returns a point
// owned by the same device for every requested checkpoint.
unsafe impl ImmediateBackend for ImmediateMock {
    fn install_waits(_: &Rc<()>, _: LaneId, _: &[StepWait]) -> WaitInstallOutcome<Self> {
        WaitInstallOutcome::Ready { issued: false }
    }
    fn close_immediate(_: Rc<()>, _: ImmediateCloseReason) -> ImmediateClose<Self> {
        ImmediateClose::Issued
    }
    fn present_immediate(
        _: &Rc<()>,
        _: LaneId,
        _: &[StepWait],
        binding: PresentBinding<Self>,
    ) -> std::result::Result<ImmediateClose<Self>, (Error, PresentBinding<Self>)> {
        Ok(ImmediateClose::Issued)
    }
    fn checkpoints(
        _: &Rc<()>,
        plan: &SubmissionPlan<Self>,
        _: &[StepId],
        request: &CompletionRequest,
    ) -> CheckpointOutcome<Self> {
        let requested = match request {
            CompletionRequest::Steps(steps) => steps,
            _ => return CheckpointOutcome::Covered(Vec::new()),
        };
        let Some(&step) = requested.last() else {
            return CheckpointOutcome::Covered(Vec::new());
        };
        let point = unsafe { CompletionPoint::from_native(plan.id().device, Rc::new(())) };
        CheckpointOutcome::Covered(vec![Checkpoint {
            point,
            covers: CompletionCoverage {
                lane_frontiers: vec![(plan.steps()[step.index as usize].lane, step)],
            },
        }])
    }
    fn emergency_checkpoint(
        _: &Rc<()>,
        _: &SubmissionPlan<Self>,
        _: &[StepId],
    ) -> CheckpointOutcome<Self> {
        CheckpointOutcome::Covered(Vec::new())
    }
    fn abandon_session(_: &Rc<()>, _: &SubmissionPlan<Self>, _: &[StepId]) {}
}

fn device<B: Backend>(native: B::NativeDevice) -> Device<B> {
    let id = DeviceId::allocate().unwrap();
    let lane = LaneId {
        device: id,
        ordinal: 0,
    };
    let mut caps = DeviceCapabilities::default();
    caps.lanes = vec![LaneInfo {
        id: lane,
        class: LaneClass::Graphics,
        routes: vec![],
    }]
    .into();
    // SAFETY: the caller uses the generated device identity for its plan lane.
    unsafe { Device::from_native(id, native, caps) }
}

fn commands_plan<B: Backend>(device: &Device<B>, count: usize, ordinal: u64) -> SubmissionPlan<B> {
    let lane = device.lanes()[0].id;
    let steps: Arc<[StepDesc]> = (0..count)
        .map(|_| StepDesc {
            lane,
            kind: StepKind::Commands,
            waits: vec![],
        })
        .collect();
    // SAFETY: this diagnostic supplies an ordered, single-lane device-local plan.
    unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal,
            },
            steps,
        )
    }
}

fn validation_samples() {
    let device = device::<ValidationMock>(Arc::new(()));
    for count in [2usize, 32, 256, 1024] {
        let plan = commands_plan(&device, count, count as u64);
        let iterations = (2_000_000 / count).max(2_000);
        for _ in 0..1_000 {
            black_box(plan.validate_for_execution(&device)).unwrap();
        }
        for sample in 0..21 {
            let start = Instant::now();
            for _ in 0..iterations {
                black_box(plan.validate_for_execution(&device)).unwrap();
            }
            let elapsed = start.elapsed().as_nanos();
            println!(
                "validation,{count},no_wait,{sample},{iterations},{elapsed},{:.3}",
                elapsed as f64 / iterations as f64
            );
        }
    }
}

fn run_immediate(
    device: &Device<ImmediateMock>,
    plan: &SubmissionPlan<ImmediateMock>,
    requested: Arc<[StepId]>,
) {
    // SAFETY: the mock has no native resource lifetime and each session is finished below.
    let mut session = unsafe {
        device
            .immediate(plan.clone(), CompletionRequest::Steps(requested))
            .unwrap()
    };
    for _ in plan.steps() {
        let encoder = match session.begin_commands() {
            Ok(encoder) => encoder,
            Err(_) => panic!("mock immediate command begin failed"),
        };
        assert!(matches!(encoder.close(), ImmediateRunOutcome::Issued(_)));
    }
    assert!(matches!(
        session.finish(),
        ImmediateFinishOutcome::Issued(_)
    ));
}

fn immediate_samples() {
    let device = device::<ImmediateMock>(Rc::new(()));
    for count in [1_000usize, 10_000] {
        let plan = commands_plan(&device, count, count as u64 + 20_000);
        let all: Arc<[StepId]> = (0..count).map(|index| plan.step(index)).collect();
        let every_16: Arc<[StepId]> = (0..count)
            .step_by(16)
            .map(|index| plan.step(index))
            .collect();
        let iterations = (20_000 / count).max(5);
        for (case, requested) in [("all_steps", all), ("every_16th", every_16)] {
            for _ in 0..3 {
                run_immediate(&device, &plan, requested.clone());
            }
            for sample in 0..21 {
                let start = Instant::now();
                for _ in 0..iterations {
                    run_immediate(&device, &plan, requested.clone());
                }
                let elapsed = start.elapsed().as_nanos();
                println!(
                    "immediate_steps,{count},{case},{sample},{iterations},{elapsed},{:.3}",
                    elapsed as f64 / iterations as f64
                );
            }
        }
    }
}

fn main() {
    println!("workload,steps,case,sample,iterations,elapsed_ns,ns_per_run");
    validation_samples();
    immediate_samples();
}
