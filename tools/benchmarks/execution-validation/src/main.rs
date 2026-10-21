//! Synthetic CPU-only diagnostic of Core plan validation; not a GPU benchmark.
use fluxel_rhi_core::api::*;
use std::{hint::black_box, sync::Arc, time::Instant};

struct Mock;
struct Lease;
// SAFETY: the diagnostic never constructs or uses a surface image.
unsafe impl NativeImageLease for Lease {
    fn try_handoff(&mut self, _: u64) -> Result<()> {
        unreachable!()
    }
    fn is_current(&self, _: u64) -> bool {
        unreachable!()
    }
}
impl Backend for Mock {
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
    type NativeSurfaceImage = Lease;
    type NativeUncertainty = ();
    type Compute = Unavailable;
    type Mesh = Unavailable;
    type Ray = Unavailable;
    type Extras = Unavailable;
}

fn main() {
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
    // SAFETY: the inert mock device and its only lane use the same identity.
    let device = unsafe { Device::<Mock>::from_native(id, Arc::new(()), caps) };
    println!("steps,case,sample,iterations,elapsed_ns,ns_per_validation");
    for count in [2usize, 32, 256, 1024] {
        let plan_id = PlanId {
            device: id,
            ordinal: count as u64,
        };
        let make_plan = |chain: bool| {
            let steps: Arc<[StepDesc]> = (0..count)
                .map(|index| StepDesc {
                    lane,
                    kind: StepKind::Commands,
                    waits: if chain && index > 0 {
                        vec![StepWait {
                            previous: StepId {
                                plan: plan_id,
                                index: (index - 1) as u32,
                            },
                            destination: AccessScope {
                                stages: StageMask::ALL_COMMANDS,
                                accesses: AccessMask::ALL,
                            },
                        }]
                    } else {
                        vec![]
                    },
                })
                .collect();
            // SAFETY: the synthetic plan is ordered, device-local and single-lane.
            unsafe { SubmissionPlan::<Mock>::from_compiled(plan_id, steps) }
        };
        let plans = [make_plan(false), make_plan(true)];
        let iterations = (2_000_000 / count).max(2_000);
        for _ in 0..1000 {
            for plan in &plans {
                black_box(black_box(plan).validate_for_execution(black_box(&device))).unwrap();
            }
        }
        for sample in 0..21 {
            for offset in 0..2 {
                let case = (sample + offset) % 2;
                let plan = &plans[case];
                let start = Instant::now();
                for _ in 0..iterations {
                    black_box(black_box(plan).validate_for_execution(black_box(&device))).unwrap();
                }
                let elapsed = start.elapsed().as_nanos();
                println!(
                    "{count},{},{sample},{iterations},{elapsed},{:.3}",
                    if case == 0 { "no_wait" } else { "chain" },
                    elapsed as f64 / iterations as f64
                );
            }
        }
    }
}
