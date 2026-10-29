//! Cross-queue GPU signal/wait and host completion evidence.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
#[test]
fn ownership_compute_direct_compute_buffer() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let direct = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let compute = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Compute)
        .unwrap()
        .id;
    let domain = d.multi_queue().unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 80,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 80,
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
                    buffer: &buffer,
                    range: 0..80,
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
    let shader = block(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Dxil,
        bytes: Arc::from(include_bytes!("assets/multi_count/count.dxil").as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: None,
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
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..80,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    let mut producer = d.encoder(compute).unwrap();
    producer
        .transition_resources(&[barrier(empty, write)])
        .unwrap();
    {
        let mut pass = d.compute().unwrap().begin(&mut producer).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    let handoff = |from, to, ownership| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..80,
            },
            usage: StateTransition { from, to },
            ownership: Some(ownership),
        })
    };
    producer
        .transition_resources(&[handoff(
            write,
            read,
            OwnershipTransfer::Release {
                from: compute,
                to: direct,
            },
        )])
        .unwrap();
    let dest = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 80,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut consumer = d.encoder(direct).unwrap();
    consumer
        .transition_resources(&[handoff(
            write,
            read,
            OwnershipTransfer::Acquire {
                from: compute,
                to: direct,
            },
        )])
        .unwrap();
    consumer
        .transition_resources(&[handoff(
            read,
            read,
            OwnershipTransfer::Release {
                from: direct,
                to: compute,
            },
        )])
        .unwrap();
    let mut final_consumer = d.encoder(compute).unwrap();
    final_consumer
        .transition_resources(&[handoff(
            read,
            read,
            OwnershipTransfer::Acquire {
                from: direct,
                to: compute,
            },
        )])
        .unwrap();
    final_consumer
        .copy_buffer(&buffer, 0, &dest, 0, 80)
        .unwrap();
    let ticket = final_consumer.readback_buffer(&dest, 0..80).unwrap();
    let mkplan = |ordinal, lane| unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let pp = mkplan(98901, compute);
    let cp = mkplan(98902, direct);
    let fp = mkplan(98903, compute);
    let ps = pp.step(0);
    let cs = cp.step(0);
    let fs = fp.step(0);
    let producer_packet = MultiQueuePacket {
        work: SubmitPacket {
            plan: pp,
            steps: vec![BoundStep {
                step: ps,
                payload: StepPayload::Commands(vec![producer.finish().unwrap()]),
            }],
            completion: CompletionRequest::None,
        },
        signals: vec![GpuSignalRequest {
            producer: ps,
            consumer: cs,
            consumer_lane: direct,
        }],
        waits: vec![],
    };
    // SAFETY: packets use distinct plans and the token is consumed by its designated consumer.
    let signal = match unsafe { domain.submit(producer_packet) } {
        MultiQueueSubmitOutcome::Accepted(r) => r.signals.into_iter().next().unwrap(),
        _ => panic!("producer rejected"),
    };
    let consumer_packet = MultiQueuePacket {
        work: SubmitPacket {
            plan: cp,
            steps: vec![BoundStep {
                step: cs,
                payload: StepPayload::Commands(vec![consumer.finish().unwrap()]),
            }],
            completion: CompletionRequest::None,
        },
        signals: vec![GpuSignalRequest {
            producer: cs,
            consumer: fs,
            consumer_lane: compute,
        }],
        waits: vec![GpuWait {
            signal,
            destination: AccessScope {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        }],
    };
    let signal = match unsafe { domain.submit(consumer_packet) } {
        MultiQueueSubmitOutcome::Accepted(r) => r.signals.into_iter().next().unwrap(),
        _ => panic!("direct consumer rejected"),
    };
    let final_packet = MultiQueuePacket {
        work: SubmitPacket {
            plan: fp,
            steps: vec![BoundStep {
                step: fs,
                payload: StepPayload::Commands(vec![final_consumer.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        },
        signals: vec![],
        waits: vec![GpuWait {
            signal,
            destination: AccessScope {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
        }],
    };
    let point = match unsafe { domain.submit(final_packet) } {
        MultiQueueSubmitOutcome::Accepted(r) => {
            r.work.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("final consumer rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    for (i, expected) in [0u32, 1, 2, 9].into_iter().enumerate() {
        assert_eq!(
            u32::from_ne_bytes(bytes[64 + i * 4..68 + i * 4].try_into().unwrap()),
            expected
        );
    }
}
