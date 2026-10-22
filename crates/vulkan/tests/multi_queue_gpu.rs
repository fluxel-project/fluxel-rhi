//! Cross-queue GPU signal/wait and host completion evidence.
#![cfg(feature = "hardware-tests")]
use fluxel_rhi_core::api::*;
use fluxel_rhi_vulkan::VulkanInstance;
use std::{future::Future, sync::Arc};
fn block<F: Future>(f: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        match f.as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => return v,
            std::task::Poll::Pending => std::thread::sleep(std::time::Duration::from_millis(1)),
        }
    }
}
#[test]
fn compute_producer_direct_consumer() {
    let d = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    if d.multi_queue().is_none() {
        eprintln!("Vulkan multi-queue capability unavailable; skipping");
        return;
    }
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
        format: ShaderFormat::SpirV,
        bytes: Arc::from(include_bytes!("assets/multi_queue/count.spv").as_slice()),
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
    {
        let mut pass = d.compute().unwrap().begin(&mut producer).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    producer
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..80,
            },
            usage: StateTransition {
                from: write,
                to: read,
            },
            ownership: Some(OwnershipTransfer::Release {
                from: compute,
                to: direct,
            }),
        })])
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
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..80,
            },
            usage: StateTransition {
                from: write,
                to: read,
            },
            ownership: Some(OwnershipTransfer::Acquire {
                from: compute,
                to: direct,
            }),
        })])
        .unwrap();
    consumer.copy_buffer(&buffer, 0, &dest, 0, 80).unwrap();
    let ticket = consumer.readback_buffer(&dest, 0..80).unwrap();
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
    let ps = pp.step(0);
    let cs = cp.step(0);
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
    let point = match unsafe { domain.submit(consumer_packet) } {
        MultiQueueSubmitOutcome::Accepted(r) => {
            r.work.checkpoints.into_iter().next().unwrap().point
        }
        _ => panic!("consumer rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    for (i, expected) in [0u32, 1, 2, 9].into_iter().enumerate() {
        assert_eq!(
            u32::from_ne_bytes(bytes[64 + i * 4..68 + i * 4].try_into().unwrap()),
            expected
        );
    }
    // A single ordinary packet must honor its explicit cross-lane StepWait.
    let mut producer = d.encoder(compute).unwrap();
    producer
        .transition_resources(&[barrier(read, write)])
        .unwrap();
    {
        let mut pass = d.compute().unwrap().begin(&mut producer).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    producer
        .transition_resources(&[barrier(write, read)])
        .unwrap();
    let mut consumer = d.encoder(direct).unwrap();
    consumer.copy_buffer(&buffer, 0, &dest, 0, 80).unwrap();
    let ticket = consumer.readback_buffer(&dest, 0..80).unwrap();
    let id = PlanId {
        device: d.id(),
        ordinal: 98903,
    };
    let predecessor = StepId { plan: id, index: 0 };
    // SAFETY: explicit predecessor establishes queue ordering for this plan.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            id,
            Arc::from([
                StepDesc {
                    lane: compute,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane: direct,
                    kind: StepKind::Commands,
                    waits: vec![StepWait {
                        previous: predecessor,
                        destination: AccessScope {
                            stages: StageMask::COPY,
                            accesses: AccessMask::COPY_READ,
                        },
                    }],
                },
            ]),
        )
    };
    let a = plan.step(0);
    let b = plan.step(1);
    // SAFETY: StepWait covers the producer's writes before direct copy.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![
                BoundStep {
                    step: a,
                    payload: StepPayload::Commands(vec![producer.finish().unwrap()]),
                },
                BoundStep {
                    step: b,
                    payload: StepPayload::Commands(vec![consumer.finish().unwrap()]),
                },
            ],
            completion: CompletionRequest::Steps(Arc::from([a, b])),
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => {
            assert_eq!(r.checkpoints.len(), 2);
            r.checkpoints.into_iter().last().unwrap().point
        }
        _ => panic!("ordinary cross-lane packet rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    for (i, expected) in [0u32, 1, 2, 9].into_iter().enumerate() {
        assert_eq!(
            u32::from_ne_bytes(bytes[64 + i * 4..68 + i * 4].try_into().unwrap()),
            expected
        );
    }
    block(d.wait_idle()).unwrap();
}

#[test]
fn compute_timestamps_and_cross_lane_reuse() {
    let d = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    if d.multi_queue().is_none() {
        return;
    }
    let compute = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Compute)
        .unwrap()
        .id;
    let direct = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let q = d.queries(QueryKind::Timestamp).unwrap();
    assert!(q.timestamp_frequency(compute).unwrap() > 0);
    let set = q
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 2,
        })
        .unwrap();
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::QUERY_RESOLVE | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut e = d.encoder(compute).unwrap();
    q.timestamp(&mut e, &set, 0).unwrap();
    q.timestamp(&mut e, &set, 1).unwrap();
    q.resolve(&mut e, &set, 0..2, &dst, 0).unwrap();
    let ticket = e.readback_buffer(&dst, 0..16).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 98904,
            },
            Arc::from([StepDesc {
                lane: compute,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: a single compute list records and resolves its own timestamp slots.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("compute timestamps rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    let a = u64::from_ne_bytes(bytes[0..8].try_into().unwrap());
    let b = u64::from_ne_bytes(bytes[8..16].try_into().unwrap());
    assert!(a > 0 && b >= a);
    let mut e = d.encoder(direct).unwrap();
    q.timestamp(&mut e, &set, 0).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 98905,
            },
            Arc::from([StepDesc {
                lane: direct,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the previous use completed before this slot is reused on another lane.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("completed query slot reuse rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    // Independent query slots can be used on distinct lanes concurrently.
    let fresh = q
        .create(&QueryDesc {
            kind: QueryKind::Timestamp,
            count: 2,
        })
        .unwrap();
    let mut first = d.encoder(compute).unwrap();
    q.timestamp(&mut first, &fresh, 0).unwrap();
    let mut second = d.encoder(direct).unwrap();
    q.timestamp(&mut second, &fresh, 1).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 98906,
            },
            Arc::from([
                StepDesc {
                    lane: compute,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
                StepDesc {
                    lane: direct,
                    kind: StepKind::Commands,
                    waits: vec![],
                },
            ]),
        )
    };
    let a = plan.step(0);
    let b = plan.step(1);
    // SAFETY: each lane writes a distinct slot and the completion joins both lanes.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![
                BoundStep {
                    step: a,
                    payload: StepPayload::Commands(vec![first.finish().unwrap()]),
                },
                BoundStep {
                    step: b,
                    payload: StepPayload::Commands(vec![second.finish().unwrap()]),
                },
            ],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("independent slots rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let mut e = d.encoder(direct).unwrap();
    q.resolve(&mut e, &fresh, 0..2, &dst, 0).unwrap();
    let ticket = e.readback_buffer(&dst, 0..16).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 98907,
            },
            Arc::from([StepDesc {
                lane: direct,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: the completion above proves both writes have finished.
    let outcome = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match outcome {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("cross-lane query resolve rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    assert_ne!(u64::from_ne_bytes(bytes[0..8].try_into().unwrap()), 0);
    assert_ne!(u64::from_ne_bytes(bytes[8..16].try_into().unwrap()), 0);
}
