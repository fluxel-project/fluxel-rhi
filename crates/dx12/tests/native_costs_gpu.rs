//! Native call attribution workloads; run in release on an idle Windows GPU.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{Dx12, Dx12Instance};
use std::{sync::Arc, time::Instant};

#[path = "support/mod.rs"]
mod support;
use support::block;
#[derive(Clone, Copy, Debug)]
enum Case {
    Steps,
    Lists,
    Upload,
    Bind,
}
static NEXT_PLAN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(777_019);
fn run(d: &Device<Dx12>, lane: LaneId, case: Case) -> (u128, u128, [u64; 8]) {
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 128_000,
            usage: BufferUsage::COPY_DST | BufferUsage::COPY_SRC | BufferUsage::STORAGE,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let binding = if matches!(case, Case::Bind) {
        let layout = d
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: None,
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 4,
                        access: StorageAccess::ReadWrite,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                }],
            })
            .unwrap();
        let group = d
            .create_bind_group(&BindGroupDesc {
                label: None,
                layout: &layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::StorageBufferView {
                        buffer: &output,
                        range: 0..4,
                        view: StorageBufferView::Raw,
                    },
                }],
            })
            .unwrap();
        let pl = d
            .create_pipeline_layout(&PipelineLayoutDesc {
                label: None,
                groups: &[&layout],
                push_constants: &[],
            })
            .unwrap();
        let source = b"@group(0) @binding(0) var<storage,read_write> out: array<u32>; @compute @workgroup_size(1) fn main() { out[0] = 42u; }";
        let shader = block(d.shader(ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(&source[..]),
            entries: vec![ShaderEntryPoint {
                name: "main".into(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([1, 1, 1]),
            }],
            required_features: vec![],
        }))
        .unwrap();
        let pipeline = block(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
            label: None,
            layout: pl,
            compute: PipelineStage {
                module: shader,
                entry: "main".into(),
                constants: vec![],
            },
            cache: None,
            required_subgroup_size: None,
        }))
        .unwrap();
        Some((group, pipeline))
    } else {
        None
    };
    let initial = fluxel_rhi_dx12::diagnostics::snapshot();
    let start = Instant::now();
    let mut works = Vec::new();
    let mut ticket = None;
    if matches!(case, Case::Steps | Case::Lists) {
        for i in 0..500 {
            let mut e = d.encoder(lane).unwrap();
            if i == 499 {
                e.write_buffer(&output, 0, &42u32.to_le_bytes()).unwrap();
            }
            if i == 499 {
                ticket = Some(e.readback_buffer(&output, 0..4).unwrap());
            }
            works.push(e.finish().unwrap());
        }
    } else {
        let mut e = d.encoder(lane).unwrap();
        if matches!(case, Case::Upload) {
            for i in 0..500 {
                e.write_buffer(&output, i * 256, &[42u8; 256]).unwrap();
            }
        } else {
            let (group, pipeline) = binding.as_ref().unwrap();
            e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &output,
                    range: 0..128_000,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    to: BufferUses {
                        stages: StageMask::COMPUTE,
                        accesses: AccessMask::SHADER_WRITE,
                    },
                },
                ownership: None,
            })])
            .unwrap();
            let mut pass = d.compute().unwrap().begin(&mut e).unwrap();
            pass.set_pipeline(&pipeline).unwrap();
            for _ in 0..500 {
                pass.set_bind_group(0, &group, &[]).unwrap();
            }
            pass.dispatch([1, 1, 1]).unwrap();
            pass.finish().unwrap();
            e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &output,
                    range: 0..128_000,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::COMPUTE,
                        accesses: AccessMask::SHADER_WRITE,
                    },
                    to: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                },
                ownership: None,
            })])
            .unwrap();
        }
        ticket = Some(
            e.readback_buffer(
                &output,
                0..if matches!(case, Case::Upload) {
                    128_000
                } else {
                    4
                },
            )
            .unwrap(),
        );
        works.push(e.finish().unwrap());
    }
    let ticket = ticket.unwrap();
    let step_count = if matches!(case, Case::Steps) { 500 } else { 1 };
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: NEXT_PLAN.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            },
            Arc::from(
                (0..step_count)
                    .map(|_| StepDesc {
                        lane,
                        kind: StepKind::Commands,
                        waits: vec![],
                    })
                    .collect::<Vec<_>>(),
            ),
        )
    };
    let steps = if step_count == 1 {
        vec![BoundStep {
            step: plan.step(0),
            payload: StepPayload::Commands(works),
        }]
    } else {
        works
            .into_iter()
            .enumerate()
            .map(|(i, w)| BoundStep {
                step: plan.step(i),
                payload: StepPayload::Commands(vec![w]),
            })
            .collect()
    };
    let record = start.elapsed().as_nanos();
    let submit_start = Instant::now();
    let result = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps,
            completion: CompletionRequest::End,
        })
    };
    let submit = submit_start.elapsed().as_nanos();
    let counters = fluxel_rhi_dx12::diagnostics::snapshot();
    let delta = [
        counters.queue_signals - initial.queue_signals,
        counters.execute_calls - initial.execute_calls,
        counters.submitted_lists - initial.submitted_lists,
        counters.upload_resources - initial.upload_resources,
        counters.upload_maps - initial.upload_maps,
        counters.descriptor_heap_sets - initial.descriptor_heap_sets,
        counters.upload_bytes - initial.upload_bytes,
        counters.upload_capacity_bytes - initial.upload_capacity_bytes,
    ];
    let point = match result {
        SubmitOutcome::Accepted(r) => {
            assert_eq!(r.accepted.len(), step_count);
            let checkpoint = r.checkpoints.into_iter().next().unwrap();
            assert_eq!(checkpoint.covers.lane_frontiers.len(), 1);
            assert_eq!(checkpoint.covers.lane_frontiers[0].0, lane);
            assert_eq!(
                checkpoint.covers.lane_frontiers[0].1.index as usize,
                step_count - 1
            );
            checkpoint.point
        }
        _ => panic!("submission rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    let expected = if matches!(case, Case::Upload) {
        [42u8; 4]
    } else {
        42u32.to_le_bytes()
    };
    assert_eq!(&bytes[..4], &expected);
    if matches!(case, Case::Upload) {
        assert!(bytes.iter().all(|&value| value == 42));
    }
    (record, submit, delta)
}
#[test]
fn native_costs() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let smoke = std::env::var_os("FLUXEL_NATIVE_COSTS_SMOKE").is_some();
    let (warmups, count) = if smoke { (0, 1) } else { (3, 9) };
    for case in [Case::Steps, Case::Lists, Case::Upload, Case::Bind] {
        for _ in 0..warmups {
            run(&d, lane, case);
        }
        let mut samples = Vec::new();
        for _ in 0..count {
            samples.push(run(&d, lane, case));
        }
        let mut record: Vec<_> = samples.iter().map(|s| s.0).collect();
        record.sort_unstable();
        let mut submit: Vec<_> = samples.iter().map(|s| s.1).collect();
        submit.sort_unstable();
        println!(
            "{case:?} record_ns median={} range={}..{} submit_ns median={} range={}..{} counters={:?}",
            record[count / 2],
            record[0],
            record[count - 1],
            submit[count / 2],
            submit[0],
            submit[count - 1],
            samples.iter().map(|s| s.2).collect::<Vec<_>>()
        );
    }
}
