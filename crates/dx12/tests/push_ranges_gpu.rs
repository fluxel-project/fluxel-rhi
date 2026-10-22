//! Real GPU coverage for storage descriptor arrays and partially bound holes.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
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
fn disjoint_compute_push_ranges() {
    run(false);
    run(true);
}
fn run(nonzero: bool) {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    if !d.capabilities().binding_arrays {
        return;
    }
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let out = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 2,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 8,
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
                binding: 2,
                resource: BindingResource::StorageBufferView {
                    buffer: &out,
                    range: 0..8,
                    view: StorageBufferView::Raw,
                },
            }],
        })
        .unwrap();
    let pl = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[
                PushConstantRange {
                    stages: vec![ShaderStage::Compute, ShaderStage::Vertex],
                    range: if nonzero { 8..12 } else { 0..4 },
                },
                PushConstantRange {
                    stages: vec![ShaderStage::Compute],
                    range: if nonzero { 12..16 } else { 8..12 },
                },
            ],
        })
        .unwrap();
    assert!(matches!(
        d.create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[
                PushConstantRange {
                    stages: vec![ShaderStage::Compute],
                    range: 0..8
                },
                PushConstantRange {
                    stages: vec![ShaderStage::Vertex],
                    range: 4..12
                },
            ],
        }),
        Ok(_)
    ));
    assert!(matches!(
        d.create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[PushConstantRange {
                stages: vec![ShaderStage::Compute],
                range: 0..132
            },],
        }),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        d.create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[PushConstantRange {
                stages: vec![ShaderStage::Vertex, ShaderStage::Fragment],
                range: 0..128
            },],
        }),
        Err(Error::Unsupported(_))
    ));
    let shader = block(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Dxil,
        bytes: if nonzero {
            Arc::from(include_bytes!("assets/push_nonzero.dxil").as_slice())
        } else {
            Arc::from(include_bytes!("assets/push_ranges.dxil").as_slice())
        },
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
    {
        let mut invalid = d.encoder(lane).unwrap();
        let mut pass = d.compute().unwrap().begin(&mut invalid).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        assert!(matches!(
            pass.push_constants(4, &1u32.to_ne_bytes()),
            Err(Error::InvalidRange)
        ));
    }
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 8,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let copy = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |buffer, size, from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer,
                range: 0..size,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    let mut e = d.encoder(lane).unwrap();
    e.transition_resources(&[barrier(&out, 8, empty, write)])
        .unwrap();
    {
        let mut p = d.compute().unwrap().begin(&mut e).unwrap();
        p.set_pipeline(&pipeline).unwrap();
        p.set_bind_group(0, &group, &[]).unwrap();
        p.push_constants(if nonzero { 8 } else { 0 }, &12u32.to_ne_bytes())
            .unwrap();
        p.push_constants(if nonzero { 12 } else { 8 }, &30u32.to_ne_bytes())
            .unwrap();
        p.dispatch([1, 1, 1]).unwrap();
        p.finish().unwrap();
    }
    e.transition_resources(&[barrier(&out, 8, write, copy)])
        .unwrap();
    e.copy_buffer(&out, 0, &dst, 0, 8).unwrap();
    let ticket = e.readback_buffer(&dst, 0..8).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 99002,
            },
            Arc::from([StepDesc {
                lane,
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
                payload: StepPayload::Commands(vec![e.finish().unwrap()]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match result {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submit rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(u32::from_ne_bytes(bytes[0..4].try_into().unwrap()), 12);
    assert_eq!(u32::from_ne_bytes(bytes[4..8].try_into().unwrap()), 30);
}
