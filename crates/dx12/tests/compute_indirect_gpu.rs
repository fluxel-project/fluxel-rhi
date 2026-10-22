//! GPU-produced dispatch arguments and dynamic compute CBV on both DX12 lanes.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{Dx12, Dx12Instance};
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
fn barrier(
    buffer: &Buffer<Dx12>,
    size: u64,
    from: BufferUses,
    to: BufferUses,
) -> ResourceTransition<'_, Dx12> {
    ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer,
            range: 0..size,
        },
        usage: StateTransition { from, to },
        ownership: None,
    })
}
#[test]
fn gpu_generated_dispatch_and_dynamic_uniform_on_both_lanes() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    assert!(d.lanes().iter().any(|l| l.class == LaneClass::Graphics));
    assert!(d.lanes().iter().any(|l| l.class == LaneClass::Compute));
    for lane in d
        .lanes()
        .iter()
        .filter(|l| matches!(l.class, LaneClass::Graphics | LaneClass::Compute))
    {
        let buf = |size, usage| {
            d.create_buffer(&BufferDesc {
                label: None,
                size,
                usage,
                memory: MemoryClass::Device,
            })
            .unwrap()
        };
        let args = buf(16, BufferUsage::STORAGE | BufferUsage::INDIRECT);
        let output = buf(12, BufferUsage::STORAGE | BufferUsage::COPY_SRC);
        let uniform = buf(512, BufferUsage::UNIFORM | BufferUsage::COPY_DST);
        let readback = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 12,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        let entry = |binding, ty| BindGroupLayoutEntry {
            binding,
            visibility: vec![ShaderStage::Compute],
            ty,
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        };
        let storage = |min_size| BindingType::StorageBuffer {
            dynamic_offset: false,
            min_size,
            access: StorageAccess::ReadWrite,
        };
        let a = d
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: None,
                entries: vec![entry(0, storage(16))],
            })
            .unwrap();
        let b = d
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: None,
                entries: vec![
                    entry(
                        0,
                        BindingType::UniformBuffer {
                            dynamic_offset: true,
                            min_size: 16,
                        },
                    ),
                    entry(1, storage(12)),
                ],
            })
            .unwrap();
        let producer_group = d
            .create_bind_group(&BindGroupDesc {
                label: None,
                layout: &a,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::StorageBufferView {
                        buffer: &args,
                        range: 0..16,
                        view: StorageBufferView::Raw,
                    },
                }],
            })
            .unwrap();
        let consumer_group = d
            .create_bind_group(&BindGroupDesc {
                label: None,
                layout: &b,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: BindingResource::Buffer {
                            buffer: &uniform,
                            range: 0..256,
                        },
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::StorageBufferView {
                            buffer: &output,
                            range: 0..12,
                            view: StorageBufferView::Raw,
                        },
                    },
                ],
            })
            .unwrap();
        let pipeline = |layout: &BindGroupLayout<_>, bytes: &'static [u8]| {
            let layout = d
                .create_pipeline_layout(&PipelineLayoutDesc {
                    label: None,
                    groups: &[layout],
                    push_constants: &[],
                })
                .unwrap();
            let shader = block(d.shader(ShaderDesc {
                label: None,
                format: ShaderFormat::Dxil,
                bytes: Arc::from(bytes),
                entries: vec![ShaderEntryPoint {
                    name: "main".into(),
                    stage: ShaderStage::Compute,
                    workgroup_size: None,
                }],
                required_features: vec![],
            }))
            .unwrap();
            block(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
                label: None,
                layout,
                compute: PipelineStage {
                    module: shader,
                    entry: "main".into(),
                    constants: vec![],
                },
                required_subgroup_size: None,
                cache: None,
            }))
            .unwrap()
        };
        let producer = pipeline(&a, include_bytes!("assets/compute_indirect/producer.dxil"));
        let consumer = pipeline(&b, include_bytes!("assets/compute_indirect/consumer.dxil"));
        for offsets in [&[][..], &[1][..], &[512][..]] {
            let mut invalid = d.encoder(lane.id).unwrap();
            let mut pass = d.compute().unwrap().begin(&mut invalid).unwrap();
            pass.set_pipeline(&consumer).unwrap();
            assert!(matches!(
                pass.set_bind_group(0, &consumer_group, offsets),
                Err(Error::InvalidRange)
            ));
        }
        for offset in [2, 8] {
            let mut invalid = d.encoder(lane.id).unwrap();
            let mut pass = d.compute().unwrap().begin(&mut invalid).unwrap();
            pass.set_pipeline(&consumer).unwrap();
            assert!(matches!(
                pass.dispatch_indirect(&args, offset),
                Err(Error::InvalidRange)
            ));
        }
        let mut e = d.encoder(lane.id).unwrap();
        let mut values = vec![0u8; 512];
        values[256..260].copy_from_slice(&42u32.to_ne_bytes());
        e.write_buffer(&uniform, 0, &values).unwrap();
        let empty = BufferUses {
            stages: StageMask::EMPTY,
            accesses: AccessMask::EMPTY,
        };
        let write = BufferUses {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::SHADER_WRITE,
        };
        let indirect = BufferUses {
            stages: StageMask::INDIRECT,
            accesses: AccessMask::INDIRECT_READ,
        };
        let read = BufferUses {
            stages: StageMask::COPY,
            accesses: AccessMask::COPY_READ,
        };
        let uniform_read = BufferUses {
            stages: StageMask::COMPUTE,
            accesses: AccessMask::UNIFORM_READ,
        };
        e.transition_resources(&[
            barrier(&args, 16, empty, write),
            barrier(&output, 12, empty, write),
            barrier(&uniform, 512, empty, uniform_read),
        ])
        .unwrap();
        {
            let mut p = d.compute().unwrap().begin(&mut e).unwrap();
            p.set_pipeline(&producer).unwrap();
            p.set_bind_group(0, &producer_group, &[]).unwrap();
            p.dispatch([1, 1, 1]).unwrap();
            p.finish().unwrap();
        }
        e.transition_resources(&[barrier(&args, 16, write, indirect)])
            .unwrap();
        {
            let mut p = d.compute().unwrap().begin(&mut e).unwrap();
            p.set_pipeline(&consumer).unwrap();
            p.set_bind_group(0, &consumer_group, &[256]).unwrap();
            p.dispatch_indirect(&args, 4).unwrap();
            p.finish().unwrap();
        }
        e.transition_resources(&[barrier(&output, 12, write, read)])
            .unwrap();
        e.copy_buffer(&output, 0, &readback, 0, 12).unwrap();
        let ticket = e.readback_buffer(&readback, 0..12).unwrap();
        let work = e.finish().unwrap();
        drop(args);
        drop(output);
        drop(uniform);
        drop(producer_group);
        drop(consumer_group);
        drop(producer);
        drop(consumer);
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal: 99600 + u64::from(lane.id.ordinal),
                },
                Arc::from([StepDesc {
                    lane: lane.id,
                    kind: StepKind::Commands,
                    waits: vec![],
                }]),
            )
        };
        let step = plan.step(0);
        let result = unsafe {
            d.queue().submit(SubmitPacket {
                plan,
                steps: vec![BoundStep {
                    step,
                    payload: StepPayload::Commands(vec![work]),
                }],
                completion: CompletionRequest::End,
            })
        };
        let point = match result {
            SubmitOutcome::Accepted(receipt) => {
                receipt.checkpoints.into_iter().next().unwrap().point
            }
            _ => panic!("submission rejected"),
        };
        block(d.wait_completion(&point)).unwrap();
        let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
        assert_eq!(
            bytes
                .chunks_exact(4)
                .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>(),
            [42, 43, 44]
        );
    }
}
