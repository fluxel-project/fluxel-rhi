//! Contract tests for device provenance, submission, and immediate execution.
//!
//! Test backends are local witnesses for Core invariants; they do not model a
//! native GPU implementation or replace backend conformance testing.
#![allow(missing_docs, dead_code, unused_variables, unused_imports)]
use super::*;
use std::{
    future::Future,
    marker::PhantomData,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[path = "tests/binding_arrays.rs"]
mod binding_arrays;

#[path = "tests/ergonomics.rs"]
mod ergonomics;
#[path = "tests/execution_contract.rs"]
mod execution_contract;
#[path = "tests/multiqueue_validation.rs"]
mod multiqueue_validation;
#[path = "tests/query_validation.rs"]
mod query_validation;
#[path = "tests/ray_validation.rs"]
pub(crate) mod ray_validation;
#[path = "tests/type_contract_lane.rs"]
mod type_contract_lane;

#[cfg(test)]
pub(crate) mod type_contract_tests {
    use super::*;
    use std::{
        cell::Cell,
        future::{Ready, ready},
        rc::Rc,
    };
    thread_local! {
        static ABANDONED_STEPS: Cell<usize> = const { Cell::new(0) };
        static DRAW_CALLS: Cell<usize> = const { Cell::new(0) };
        pub(crate) static GL_TRANSITIONS: Cell<usize> = const { Cell::new(0) };
        pub(crate) static GL_CHECKPOINTS: Cell<usize> = const { Cell::new(0) };
        pub(crate) static GL_EMPTY_STEP_CHECKPOINTS: Cell<bool> = const { Cell::new(false) };
        pub(crate) static GL_CLOSE_FAILURE: Cell<u8> = const { Cell::new(0) };
        pub(crate) static GL_WAIT_REJECT: Cell<bool> = const { Cell::new(false) };
        pub(crate) static GL_CHECKPOINT_UNKNOWN: Cell<bool> = const { Cell::new(false) };
        pub(crate) static GL_EMERGENCY_UNKNOWN: Cell<bool> = const { Cell::new(false) };
        pub(crate) static GL_DEVICE: Cell<Option<DeviceId>> = const { Cell::new(None) };
        pub(crate) static GL_ABANDONED: Cell<usize> = const { Cell::new(0) };
    }

    // SAFETY: the unit test lease has no native state and always accepts handoff.
    unsafe impl NativeImageLease for () {
        fn try_handoff(&mut self, _: u64) -> Result<()> {
            Ok(())
        }
        fn is_current(&self, _: u64) -> bool {
            true
        }
    }
    pub(crate) struct TestLease {
        current: Arc<AtomicU64>,
        handed_off: bool,
    }
    // SAFETY: atomic generation and `handed_off` make the test lease single-use.
    unsafe impl NativeImageLease for TestLease {
        fn try_handoff(&mut self, generation: u64) -> Result<()> {
            if !self.is_current(generation) {
                return Err(Error::SurfaceOutdated);
            }
            self.handed_off = true;
            Ok(())
        }
        fn is_current(&self, generation: u64) -> bool {
            self.current.load(Ordering::Acquire) == generation && !self.handed_off
        }
    }

    pub(crate) struct NativeShared;
    pub(crate) struct GlLocal;
    impl Backend for NativeShared {
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
        type NativeSurfaceImage = TestLease;
        type NativeUncertainty = ();
        type Compute = Unavailable;
        type Mesh = Unavailable;
        type Ray = Unavailable;
        type Extras = Unavailable;
    }
    impl Backend for GlLocal {
        type NativeDevice = Rc<()>;
        type NativeBuffer = Rc<()>;
        type NativeTexture = Rc<()>;
        type NativeTextureView = Rc<()>;
        type NativeSampler = Rc<()>;
        type NativeBindGroupLayout = Rc<()>;
        type NativePipelineLayout = Rc<()>;
        type NativeBindGroup = Rc<()>;
        type NativeShader = Rc<()>;
        type NativeRasterPipeline = Rc<()>;
        type NativeComputePipeline = Rc<()>;
        type NativeMeshPipeline = Rc<()>;
        type NativeRayPipeline = Rc<()>;
        type NativePipelineCache = Rc<()>;
        type NativeAcceleration = Rc<()>;
        type NativeSparseAllocation = Rc<()>;
        type NativeQuerySet = Rc<()>;
        type NativeEncoder = Rc<()>;
        type NativeCompletion = Rc<()>;
        type NativeSurface = Rc<()>;
        type NativeSurfaceImage = ();
        type NativeUncertainty = ();
        type Compute = Unavailable;
        type Mesh = Unavailable;
        type Ray = Unavailable;
        type Extras = Unavailable;
    }
    impl SharedBackend for NativeShared {}
    impl CommandBackend for NativeShared {
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
    impl CommandBackend for GlLocal {
        fn begin_encoder(_: &Rc<()>, _: LaneId) -> Result<Rc<()>> {
            Ok(Rc::new(()))
        }
        fn transition(_: &mut Rc<()>, _: &[ResourceTransition<'_, Self>]) -> Result<()> {
            GL_TRANSITIONS.with(|calls| calls.set(calls.get() + 1));
            Ok(())
        }
        fn copy_buffer(
            _: &mut Rc<()>,
            _: &Rc<()>,
            _: u64,
            _: &Rc<()>,
            _: u64,
            _: u64,
        ) -> Result<()> {
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
    impl RasterBackend for NativeShared {
        fn begin_raster(_: &mut (), _: &RasterPassDesc<'_, Self>) -> Result<()> {
            Ok(())
        }
        fn set_pipeline(_: &mut (), _: &Arc<()>) -> Result<()> {
            Ok(())
        }
        fn set_bind_group(_: &mut (), _: u32, _: &Arc<()>, _: &[u32]) -> Result<()> {
            Ok(())
        }
        fn set_vertex_buffer(_: &mut (), _: u32, _: &Arc<()>, _: u64, _: u64) -> Result<()> {
            Ok(())
        }
        fn set_index_buffer(_: &mut (), _: &Arc<()>, _: u64, _: u64, _: IndexFormat) -> Result<()> {
            Ok(())
        }
        fn viewport(_: &mut (), _: Viewport) -> Result<()> {
            Ok(())
        }
        fn scissor(_: &mut (), _: Rect) -> Result<()> {
            Ok(())
        }
        fn blend_constant(_: &mut (), _: [f32; 4]) -> Result<()> {
            Ok(())
        }
        fn stencil_reference(_: &mut (), _: u32) -> Result<()> {
            Ok(())
        }
        fn draw(_: &mut (), _: Range<u32>, _: Range<u32>) -> Result<()> {
            DRAW_CALLS.with(|calls| calls.set(calls.get() + 1));
            Ok(())
        }
        fn draw_indexed(_: &mut (), _: Range<u32>, _: i32, _: Range<u32>) -> Result<()> {
            DRAW_CALLS.with(|calls| calls.set(calls.get() + 1));
            Ok(())
        }
        fn end_raster(_: &mut ()) -> Result<()> {
            Ok(())
        }
        fn abort_raster(_: &mut ()) {}
    }
    impl CompletionBackend for NativeShared {
        type WaitFuture<'a> = Ready<Result<()>>;
        fn completion_status(_: &Arc<()>, _: &Arc<()>) -> CompletionState {
            CompletionState::Complete
        }
        fn wait_completion<'a>(_: &'a Arc<()>, _: &'a Arc<()>) -> Self::WaitFuture<'a> {
            ready(Ok(()))
        }
        fn wait_idle<'a>(_: &'a Arc<()>) -> Self::WaitFuture<'a> {
            ready(Ok(()))
        }
    }
    impl CompletionBackend for GlLocal {
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
    impl DeferredBackend for NativeShared {
        type NativeRecordedWork = ();
        fn finish_encoder(_: (), _: LaneId) -> Result<()> {
            Ok(())
        }
        // SAFETY: this mock never issues native work and returns the owning
        // packet unchanged; there is no accepted prefix to retain or retire.
        unsafe fn submit(_: &Arc<()>, packet: SubmitPacket<Self>) -> SubmitOutcome<Self> {
            SubmitOutcome::Rejected {
                error: Error::Unsupported("mock"),
                work: RetryableSubmit::Fresh(packet),
            }
        }
        // SAFETY: this mock also rejects a tail without touching its accepted
        // frontier, so the caller retains its original ownership and proof.
        unsafe fn resume(_: &Arc<()>, tail: DeferredTail<Self>) -> SubmitOutcome<Self> {
            SubmitOutcome::Rejected {
                error: Error::Unsupported("mock"),
                work: RetryableSubmit::Tail(tail),
            }
        }
    }
    // SAFETY: this fixture reports every issued prefix deterministically.
    unsafe impl ImmediateBackend for GlLocal {
        fn install_waits(_: &Rc<()>, _: LaneId, _: &[StepWait]) -> WaitInstallOutcome<Self> {
            if GL_WAIT_REJECT.with(|v| v.replace(false)) {
                WaitInstallOutcome::Rejected(Error::Backend("wait rejected".into()))
            } else {
                WaitInstallOutcome::Ready { issued: false }
            }
        }
        fn close_immediate(_: Rc<()>, reason: ImmediateCloseReason) -> ImmediateClose<Self> {
            if matches!(reason, ImmediateCloseReason::Aborted)
                && GL_CLOSE_FAILURE.with(|mode| mode.get()) == 3
            {
                GL_CLOSE_FAILURE.with(|mode| mode.set(0));
                return ImmediateClose::Issued;
            }
            if matches!(reason, ImmediateCloseReason::Finished) {
                match GL_CLOSE_FAILURE.with(|mode| mode.replace(0)) {
                    1 => {
                        return ImmediateClose::Failed {
                            error: Error::Backend("partial close".into()),
                            issued: true,
                        };
                    }
                    2 => {
                        return ImmediateClose::Indeterminate {
                            error: Error::Backend("uncertain close".into()),
                            uncertain: NativeUncertainty {
                                device: GL_DEVICE.with(|id| id.get().unwrap()),
                                native: (),
                            },
                        };
                    }
                    _ => {}
                }
            }
            match reason {
                ImmediateCloseReason::Finished => ImmediateClose::Issued,
                ImmediateCloseReason::Aborted => ImmediateClose::Failed {
                    error: Error::InvalidState,
                    issued: false,
                },
            }
        }
        fn present_immediate(
            _: &Rc<()>,
            _: LaneId,
            _: &[StepWait],
            _: PresentBinding<Self>,
        ) -> std::result::Result<ImmediateClose<Self>, (Error, PresentBinding<Self>)> {
            Ok(ImmediateClose::Issued)
        }
        fn checkpoints(
            _: &Rc<()>,
            plan: &SubmissionPlan<Self>,
            issued: &[StepId],
            request: &CompletionRequest,
        ) -> CheckpointOutcome<Self> {
            GL_CHECKPOINTS.with(|calls| calls.set(calls.get() + 1));
            if GL_CHECKPOINT_UNKNOWN.with(|v| v.replace(false)) {
                CheckpointOutcome::Indeterminate(NativeUncertainty {
                    device: GL_DEVICE.with(|id| id.get().unwrap()),
                    native: (),
                })
            } else if GL_EMPTY_STEP_CHECKPOINTS.with(|v| v.get())
                && matches!(request, CompletionRequest::Steps(steps) if steps.len() == 1)
            {
                CheckpointOutcome::Covered(Vec::new())
            } else if GL_EMPTY_STEP_CHECKPOINTS.with(|v| v.get())
                && matches!(request, CompletionRequest::Steps(steps) if steps.len() > 1)
            {
                let Some(&frontier) = issued.last() else {
                    return CheckpointOutcome::Covered(Vec::new());
                };
                CheckpointOutcome::Covered(vec![Checkpoint {
                    // SAFETY: this mock point belongs to the fixture device.
                    point: unsafe {
                        CompletionPoint::from_native(
                            GL_DEVICE.with(|id| id.get().unwrap()),
                            Rc::new(()),
                        )
                    },
                    covers: CompletionCoverage {
                        lane_frontiers: vec![(
                            plan.steps()[frontier.index as usize].lane,
                            frontier,
                        )],
                    },
                }])
            } else {
                CheckpointOutcome::Covered(Vec::new())
            }
        }
        fn emergency_checkpoint(
            _: &Rc<()>,
            _: &SubmissionPlan<Self>,
            _: &[StepId],
        ) -> CheckpointOutcome<Self> {
            GL_CHECKPOINTS.with(|calls| calls.set(calls.get() + 1));
            if GL_EMERGENCY_UNKNOWN.with(|v| v.replace(false)) {
                CheckpointOutcome::Indeterminate(NativeUncertainty {
                    device: GL_DEVICE.with(|id| id.get().unwrap()),
                    native: (),
                })
            } else {
                CheckpointOutcome::Covered(Vec::new())
            }
        }
        fn abandon_session(_: &Rc<()>, _: &SubmissionPlan<Self>, issued: &[StepId]) {
            ABANDONED_STEPS.with(|count| count.set(issued.len()));
            GL_ABANDONED.with(|count| count.set(count.get() + 1));
        }
    }
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_send<T: Send>() {}
    fn generic_renderer_optional_compute<B: Backend + CommandBackend>(
        device: &Device<B>,
        encoder: &mut CommandEncoder<B>,
    ) -> Result<()> {
        if let Some(compute) = device.compute() {
            let mut pass = compute.begin(encoder)?;
            pass.dispatch([1, 1, 1])?;
            pass.finish()?;
        }
        Ok(())
    }
    #[test]
    fn concrete_shared_and_local_profiles_typecheck() {
        assert_send_sync::<Device<NativeShared>>();
        assert_send_sync::<Buffer<NativeShared>>();
        assert_send_sync::<Queue<NativeShared>>();
        assert_send::<CommandEncoder<NativeShared>>();
        assert_send::<RecordedWork<NativeShared>>();
        assert_send_sync::<CompletionPoint<NativeShared>>();
        let id = DeviceId::allocate().unwrap();
        // SAFETY: this fixture creates matching native values for the allocated device ID.
        let native = unsafe {
            Device::<NativeShared>::from_native(id, Arc::new(()), DeviceCapabilities::default())
        };
        // SAFETY: this fixture creates matching native values for the allocated device ID.
        let local = unsafe {
            Device::<GlLocal>::from_native(id, Rc::new(()), DeviceCapabilities::default())
        };
        let lane = LaneId {
            device: id,
            ordinal: 0,
        };
        let mut encoder = CommandEncoder::<NativeShared>::new(&native, lane).unwrap();
        generic_renderer_optional_compute(&native, &mut encoder).unwrap();
        let _ = CommandEncoder::<GlLocal>::new(&local, lane).unwrap();
    }
    #[test]
    fn raster_draw_rejects_inverted_ranges_before_backend() {
        DRAW_CALLS.with(|calls| calls.set(0));
        let id = DeviceId::allocate().unwrap();
        // SAFETY: the fixture pairs this native device with its newly allocated ID.
        let device = unsafe {
            Device::<NativeShared>::from_native(id, Arc::new(()), DeviceCapabilities::default())
        };
        let lane = LaneId {
            device: id,
            ordinal: 0,
        };
        let mut encoder = CommandEncoder::new(&device, lane).unwrap();
        let colors = [];
        let mut pass = encoder
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &colors,
                depth_stencil: None,
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        assert!(matches!(
            pass.draw(Range { start: 4, end: 3 }, 0..1),
            Err(Error::InvalidRange)
        ));
        assert!(matches!(
            pass.draw_indexed(0..1, 0, Range { start: 2, end: 1 }),
            Err(Error::InvalidRange)
        ));
        DRAW_CALLS.with(|calls| assert_eq!(calls.get(), 0));
        pass.draw(0..3, 0..1).unwrap();
        DRAW_CALLS.with(|calls| assert_eq!(calls.get(), 1));
        pass.finish().unwrap();
    }
    #[test]
    fn clear_texture_rejects_invalid_shape_before_backend() {
        let id = DeviceId::allocate().unwrap();
        let other = DeviceId::allocate().unwrap();
        // SAFETY: each mock object is paired with its allocated device identity.
        let device = unsafe {
            Device::<NativeShared>::from_native(id, Arc::new(()), DeviceCapabilities::default())
        };
        let texture = unsafe { Texture::<NativeShared>::from_native(id, Arc::new(())) };
        let foreign = unsafe { Texture::<NativeShared>::from_native(other, Arc::new(())) };
        let mut encoder = CommandEncoder::new(
            &device,
            LaneId {
                device: id,
                ordinal: 0,
            },
        )
        .unwrap();
        let color = TextureClearValue::Color(ClearColor::Float([0.0; 4]));
        assert!(matches!(
            encoder.clear_texture(&foreign, 0..1, 0..1, color),
            Err(Error::WrongDevice)
        ));
        assert!(matches!(
            encoder.clear_texture(&texture, 0..0, 0..1, color),
            Err(Error::InvalidRange)
        ));
        assert!(matches!(
            encoder.clear_texture(&texture, 0..1, 1..1, color),
            Err(Error::InvalidRange)
        ));
        assert!(matches!(
            encoder.clear_texture(&texture, 0..1, 0..1, TextureClearValue::Depth(f32::NAN)),
            Err(Error::InvalidRange)
        ));
        assert!(matches!(
            encoder.clear_texture(&texture, 0..1, 0..1, TextureClearValue::Depth(1.1)),
            Err(Error::InvalidRange)
        ));
        assert!(matches!(
            encoder.clear_texture(&texture, 0..1, 0..1, color),
            Err(Error::Unsupported(_))
        ));
    }
    #[test]
    fn native_recording_can_be_borrowed_without_consumption() {
        let device = DeviceId::allocate().unwrap();
        let work = RecordedWork::<NativeShared> {
            device,
            lane: LaneId { device, ordinal: 0 },
            native: (),
        };
        // SAFETY: this mock backend only inspects its own work and retains nothing.
        assert_eq!(unsafe { work.native_ref() }, &());
        assert_eq!(work.device(), device);
        // SAFETY: the mock backend consumes its recording exactly once.
        assert_eq!(unsafe { work.into_native() }, ());
    }
    #[test]
    fn deferred_binding_rejects_wrong_lane_and_foreign_tail() {
        let device_id = DeviceId::allocate().unwrap();
        let other_id = DeviceId::allocate().unwrap();
        // SAFETY: the fixture pairs this native device with its newly allocated ID.
        let device = unsafe {
            Device::<NativeShared>::from_native(
                device_id,
                Arc::new(()),
                DeviceCapabilities::default(),
            )
        };
        let lane = LaneId {
            device: device_id,
            ordinal: 0,
        };
        let wrong_lane = LaneId {
            device: device_id,
            ordinal: 1,
        };
        // SAFETY: this test provides an ordered, device-local compiled plan.
        let plan = unsafe {
            SubmissionPlan::<NativeShared>::from_compiled(
                PlanId {
                    device: device_id,
                    ordinal: 1,
                },
                Arc::from([StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: Vec::new(),
                }]),
            )
        };
        let packet = SubmitPacket {
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Commands(vec![RecordedWork {
                    device: device_id,
                    lane: wrong_lane,
                    native: (),
                }]),
            }],
            plan: plan.clone(),
            completion: CompletionRequest::End,
        };
        // SAFETY: the test intentionally invokes submit to assert Core rejects the invalid lane.
        match unsafe { device.queue().submit(packet) } {
            SubmitOutcome::Rejected {
                error: Error::WrongLane,
                ..
            } => {}
            _ => panic!("wrong-lane work reached the backend"),
        }
        // SAFETY: this test constructs a device-local plan for the other fixture device.
        let foreign_plan = unsafe {
            SubmissionPlan::<NativeShared>::from_compiled(
                PlanId {
                    device: other_id,
                    ordinal: 1,
                },
                Arc::from([]),
            )
        };
        let tail = DeferredTail {
            plan: foreign_plan,
            next_step: 0,
            pending: Vec::new(),
            accepted_frontier: Vec::new(),
            completion: CompletionRequest::None,
        };
        // SAFETY: the test intentionally invokes resume to assert Core rejects the foreign tail.
        match unsafe { device.queue().resume(tail) } {
            SubmitOutcome::Rejected {
                error: Error::WrongDevice,
                ..
            } => {}
            _ => panic!("foreign tail reached the backend"),
        }
    }
    #[test]
    fn stale_surface_lease_cannot_enter_plan_present() {
        let device_id = DeviceId::allocate().unwrap();
        // SAFETY: the fixture pairs this native device with its newly allocated ID.
        let device = unsafe {
            Device::<NativeShared>::from_native(
                device_id,
                Arc::new(()),
                DeviceCapabilities::default(),
            )
        };
        let lane = LaneId {
            device: device_id,
            ordinal: 0,
        };
        // SAFETY: this test provides an ordered, device-local compiled plan.
        let plan = unsafe {
            SubmissionPlan::<NativeShared>::from_compiled(
                PlanId {
                    device: device_id,
                    ordinal: 2,
                },
                Arc::from([StepDesc {
                    lane,
                    kind: StepKind::Present,
                    waits: Vec::new(),
                }]),
            )
        };
        let current = Arc::new(AtomicU64::new(1));
        let image = SurfaceImage {
            device: device_id,
            surface: SurfaceId::allocate().unwrap(),
            generation: 1,
            native: TestLease {
                current: current.clone(),
                handed_off: false,
            },
        };
        current.store(2, Ordering::Release);
        let packet = SubmitPacket {
            steps: vec![BoundStep {
                step: plan.step(0),
                payload: StepPayload::Present(PresentBinding::new(image)),
            }],
            plan,
            completion: CompletionRequest::None,
        };
        // SAFETY: the test invokes submit only to exercise stale-lease validation.
        match unsafe { device.queue().submit(packet) } {
            SubmitOutcome::Rejected {
                error: Error::SurfaceOutdated,
                ..
            } => {}
            _ => panic!("stale surface lease reached the backend"),
        }
    }
    #[test]
    fn immediate_validates_waits_and_lanes_before_execution() {
        let id = DeviceId::allocate().unwrap();
        let lane = LaneId {
            device: id,
            ordinal: 0,
        };
        let info = LaneInfo {
            id: lane,
            class: LaneClass::Graphics,
            routes: Vec::new(),
        };
        assert_eq!(info.route_to(lane), WaitRoute::SameQueue);
        assert_eq!(
            info.route_to(LaneId {
                device: id,
                ordinal: 1
            }),
            WaitRoute::Unsupported
        );
        let caps = DeviceCapabilities {
            lanes: Arc::from([info]),
            ..DeviceCapabilities::default()
        };
        // SAFETY: the fixture owns the native witness and its matching ID.
        let device = unsafe { Device::<GlLocal>::from_native(id, Rc::new(()), caps) };
        let plan_id = PlanId {
            device: id,
            ordinal: 19,
        };
        let scope = AccessScope {
            stages: StageMask::ALL_COMMANDS,
            accesses: AccessMask::ALL,
        };
        let make_plan = |second_lane, previous| {
            // SAFETY: malformed fixtures intentionally exercise Core validation only.
            unsafe {
                SubmissionPlan::from_compiled(
                    plan_id,
                    Arc::from([
                        StepDesc {
                            lane,
                            kind: StepKind::Commands,
                            waits: vec![],
                        },
                        StepDesc {
                            lane: second_lane,
                            kind: StepKind::Commands,
                            waits: vec![StepWait {
                                previous,
                                destination: scope,
                            }],
                        },
                    ]),
                )
            }
        };
        let valid = make_plan(
            lane,
            StepId {
                plan: plan_id,
                index: 0,
            },
        );
        // SAFETY: mock session has no GPU resources and both steps are closed.
        let mut session = unsafe { device.immediate(valid, CompletionRequest::None) }.unwrap();
        for _ in 0..2 {
            let encoder = session.begin_commands().ok().unwrap();
            assert!(matches!(encoder.close(), ImmediateRunOutcome::Issued(_)));
        }
        assert!(matches!(
            session.finish(),
            ImmediateFinishOutcome::Issued(_)
        ));
        let forward = make_plan(
            lane,
            StepId {
                plan: plan_id,
                index: 1,
            },
        );
        // SAFETY: rejected before issuing native work.
        assert!(matches!(
            unsafe { device.immediate(forward, CompletionRequest::None) },
            Err(Error::InvalidState)
        ));
        let other_lane = make_plan(
            LaneId {
                device: id,
                ordinal: 1,
            },
            StepId {
                plan: plan_id,
                index: 0,
            },
        );
        // SAFETY: rejected before issuing native work.
        assert!(matches!(
            unsafe { device.immediate(other_lane, CompletionRequest::None) },
            Err(Error::WrongLane)
        ));
    }
    #[test]
    fn dropped_immediate_session_reports_issued_prefix_to_backend() {
        ABANDONED_STEPS.with(|count| count.set(0));
        let device_id = DeviceId::allocate().unwrap();
        let lane = LaneId {
            device: device_id,
            ordinal: 0,
        };
        let caps = DeviceCapabilities {
            lanes: Arc::from([LaneInfo {
                id: lane,
                class: LaneClass::Graphics,
                routes: Vec::new(),
            }]),
            ..DeviceCapabilities::default()
        };
        // SAFETY: the fixture pairs this native device with its newly allocated ID.
        let device = unsafe { Device::<GlLocal>::from_native(device_id, Rc::new(()), caps) };
        // SAFETY: this test provides an ordered, device-local compiled plan.
        let plan = unsafe {
            SubmissionPlan::<GlLocal>::from_compiled(
                PlanId {
                    device: device_id,
                    ordinal: 3,
                },
                Arc::from([StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: Vec::new(),
                }]),
            )
        };
        // SAFETY: the fixture plan belongs to this immediate-only device.
        let mut session = unsafe { device.immediate(plan, CompletionRequest::None).unwrap() };
        let encoder = match session.begin_commands() {
            Ok(encoder) => encoder,
            Err(_) => panic!("mock command slot could not begin"),
        };
        match encoder.close() {
            ImmediateRunOutcome::Issued(_) => {}
            _ => panic!("mock command slot did not issue"),
        }
        drop(session);
        ABANDONED_STEPS.with(|count| assert_eq!(count.get(), 1));
    }
}
