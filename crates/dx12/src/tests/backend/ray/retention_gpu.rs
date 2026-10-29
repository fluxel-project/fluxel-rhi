//! Real-DXR ownership regression coverage for mutable TLAS storage.
use super::*;
use crate::Dx12Instance;
use fluxel_rhi_core::api::*;
use std::sync::Arc;

#[path = "../support.rs"]
mod support;
use support::block;

fn submit(
    device: &Device<Dx12>,
    lane: LaneId,
    work: RecordedWork<Dx12>,
    ordinal: u64,
) -> CompletionPoint<Dx12> {
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    match unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    } {
        SubmitOutcome::Accepted(mut receipt) => receipt.checkpoints.remove(0).point,
        _ => panic!("DXR lifecycle submission failed"),
    }
}

fn usage() -> BufferUses {
    BufferUses {
        stages: StageMask::ACCELERATION,
        accesses: AccessMask::ACCEL_WRITE,
    }
}

fn transition<'a>(
    buffer: &'a Buffer<Dx12>,
    size: u64,
    to: BufferUses,
) -> ResourceTransition<'a, Dx12> {
    ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer,
            range: 0..size,
        },
        usage: StateTransition {
            from: BufferUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to,
        },
        ownership: None,
    })
}

#[test]
fn tlas_commits_only_executed_builds_and_retires_rebuilt_blas() {
    let device = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    if !device.capabilities().ray {
        eprintln!("SKIP: DXR unavailable");
        return;
    }
    let lane = device
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let ray = device.ray().unwrap();
    let vertices = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 36,
            usage: BufferUsage::ACCEL_BUILD_INPUT | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let bytes: Vec<_> = [-1f32, -1., 0., 1., -1., 0., 0., 1., 0.]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
    let make_blas = || AccelerationDesc {
        input: AccelerationInput::Blas(vec![AccelerationGeometry::Triangles {
            vertices: vertices.clone(),
            vertex_offset: 0,
            vertex_count: 3,
            vertex_format: VertexFormat::Float32(3),
            stride: 12,
            indices: None,
            opaque: true,
        }]),
        fast_trace: true,
        allow_update: false,
    };
    let blas_a_desc = make_blas();
    let blas_a = ray.create_acceleration(&blas_a_desc).unwrap();
    let tlas_desc = |blas: &AccelerationStructure<Dx12>| AccelerationDesc {
        input: AccelerationInput::Tlas(vec![AccelerationInstance {
            blas: blas.clone(),
            transform: [[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]],
            mask: 255,
            custom_index: 0,
            shader_table_offset: 0,
            flags: AccelerationInstanceFlags::TRIANGLE_CULL_DISABLE,
        }]),
        fast_trace: true,
        allow_update: false,
    };
    let first_tlas_desc = tlas_desc(&blas_a);
    let tlas = ray.create_acceleration(&first_tlas_desc).unwrap();
    let bscratch_size = ray.sizes(&blas_a_desc).unwrap().scratch_bytes;
    let tscratch_size = ray.sizes(&first_tlas_desc).unwrap().scratch_bytes;
    let bscratch = device
        .create_buffer(&BufferDesc {
            label: None,
            size: bscratch_size,
            usage: BufferUsage::ACCEL_SCRATCH,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let tscratch = device
        .create_buffer(&BufferDesc {
            label: None,
            size: tscratch_size,
            usage: BufferUsage::ACCEL_SCRATCH,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let record = |blas: &AccelerationStructure<Dx12>,
                  blas_desc: &AccelerationDesc<Dx12>,
                  tlas_desc: &AccelerationDesc<Dx12>| {
        let mut encoder = device.encoder(lane).unwrap();
        encoder.write_buffer(&vertices, 0, &bytes).unwrap();
        encoder
            .transition_resources(&[
                transition(
                    &vertices,
                    36,
                    BufferUses {
                        stages: StageMask::ACCELERATION,
                        accesses: AccessMask::ACCEL_READ,
                    },
                ),
                transition(&bscratch, bscratch_size, usage()),
                transition(&tscratch, tscratch_size, usage()),
            ])
            .unwrap();
        ray.build(&mut encoder, blas, &bscratch, blas_desc).unwrap();
        encoder
            .transition_resources(&[ResourceTransition::Acceleration {
                target: blas,
                before: AccessScope {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_WRITE,
                },
                after: AccessScope {
                    stages: StageMask::ACCELERATION,
                    accesses: AccessMask::ACCEL_READ,
                },
            }])
            .unwrap();
        ray.build(&mut encoder, &tlas, &tscratch, tlas_desc)
            .unwrap();
        encoder.finish().unwrap()
    };
    let point = submit(
        &device,
        lane,
        record(&blas_a, &blas_a_desc, &first_tlas_desc),
        91_001,
    );
    block(device.wait_completion(&point)).unwrap();
    let state = unsafe { tlas.native_ref() }.0.clone();
    assert_eq!(state.committed_blas.lock().unwrap().len(), 1);
    drop(first_tlas_desc);
    let blas_b_desc = make_blas();
    let blas_b = ray.create_acceleration(&blas_b_desc).unwrap();
    let weak_a = Arc::downgrade(&unsafe { blas_a.native_ref() }.0);
    let cancelled = record(&blas_b, &blas_b_desc, &tlas_desc(&blas_b));
    drop(cancelled);
    assert!(Arc::ptr_eq(
        &state.committed_blas.lock().unwrap()[0].0,
        &weak_a.upgrade().unwrap()
    ));
    let mut previous = blas_b;
    let mut previous_desc = blas_b_desc;
    drop(blas_a);
    for round in 0..32u64 {
        let next_desc = make_blas();
        let next = ray.create_acceleration(&next_desc).unwrap();
        let next_tlas_desc = tlas_desc(&next);
        let previous_weak = Arc::downgrade(&unsafe { previous.native_ref() }.0);
        let point = submit(
            &device,
            lane,
            record(&next, &next_desc, &next_tlas_desc),
            91_002 + round,
        );
        block(device.wait_completion(&point)).unwrap();
        drop(next_tlas_desc);
        drop(previous_desc);
        drop(previous);
        assert!(
            previous_weak.upgrade().is_none(),
            "round {round} retained old BLAS"
        );
        assert_eq!(state.committed_blas.lock().unwrap().len(), 1);
        previous = next;
        previous_desc = next_desc;
    }
    assert!(weak_a.upgrade().is_none());
}
