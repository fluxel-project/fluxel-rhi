//! Upload mapping, unaligned logical uniforms and device-buffer readback.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
#[test]
fn mapped_upload_to_odd_uniform_and_direct_device_readback() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let upload = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 20,
            memory: MemoryClass::Upload,
            usage: BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
        })
        .unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 20,
            memory: MemoryClass::Device,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        })
        .unwrap();
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 4,
            memory: MemoryClass::Device,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
        })
        .unwrap();
    {
        let mut mapped = unsafe { block(d.map_buffer(&upload, 0..20, MapMode::Write)).unwrap() };
        mapped.bytes_mut().unwrap().copy_from_slice(&[42; 20]);
        mapped.flush(0..20).unwrap();
        assert!(unsafe { block(d.map_buffer(&upload, 0..4, MapMode::Write)) }.is_err());
    }
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::UniformBuffer {
                        dynamic_offset: false,
                        min_size: 20,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 4,
                        access: StorageAccess::ReadWrite,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })
        .unwrap();
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer {
                        buffer: &buffer,
                        range: 0..20,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::StorageBufferView {
                        buffer: &output,
                        range: 0..4,
                        view: StorageBufferView::Raw,
                    },
                },
            ],
        })
        .unwrap();
    let pipeline_layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = block(d.shader(ShaderDesc { label: None, format: ShaderFormat::Wgsl, bytes: Arc::from(&b"struct U { a: vec4<u32>, b: u32 }; @group(0) @binding(0) var<uniform> u: U; @group(0) @binding(1) var<storage,read_write> out: array<u32>; @compute @workgroup_size(1) fn main() { out[0] = u.b; }"[..]), entries: vec![ShaderEntryPoint { name: "main".into(), stage: ShaderStage::Compute, workgroup_size: Some([1,1,1]) }], required_features: vec![] })).unwrap();
    let pipeline = block(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
        label: None,
        layout: pipeline_layout,
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
    let copy_write = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_WRITE,
    };
    let copy_read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let uniform_read = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::UNIFORM_READ,
    };
    let shader_write = BufferUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let mut e = d.encoder(lane).unwrap();
    e.transition_resources(&[
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &upload,
                range: 0..20,
            },
            usage: StateTransition {
                from: empty,
                to: copy_read,
            },
            ownership: None,
        }),
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..20,
            },
            usage: StateTransition {
                from: empty,
                to: copy_write,
            },
            ownership: None,
        }),
    ])
    .unwrap();
    e.copy_buffer(&upload, 0, &buffer, 0, 20).unwrap();
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &buffer,
            range: 0..20,
        },
        usage: StateTransition {
            from: copy_write,
            to: uniform_read,
        },
        ownership: None,
    })])
    .unwrap();
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &output,
            range: 0..4,
        },
        usage: StateTransition {
            from: empty,
            to: shader_write,
        },
        ownership: None,
    })])
    .unwrap();
    {
        let mut p = d.compute().unwrap().begin(&mut e).unwrap();
        p.set_pipeline(&pipeline).unwrap();
        p.set_bind_group(0, &group, &[]).unwrap();
        p.dispatch([1, 1, 1]).unwrap();
        p.finish().unwrap();
    }
    e.transition_resources(&[
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..4,
            },
            usage: StateTransition {
                from: shader_write,
                to: copy_read,
            },
            ownership: None,
        }),
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &buffer,
                range: 0..20,
            },
            usage: StateTransition {
                from: uniform_read,
                to: copy_read,
            },
            ownership: None,
        }),
    ])
    .unwrap();
    let output_ticket = e.readback_buffer(&output, 0..4).unwrap();
    let ticket = e.readback_buffer(&buffer, 3..18).unwrap();
    let work = e.finish().unwrap();
    drop(upload);
    drop(buffer);
    drop(group);
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 91801,
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
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match result {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(bytes, vec![42; 15]);
    let output_bytes = unsafe { block(d.read_after(output_ticket, &point)).unwrap().bytes };
    assert_eq!(output_bytes, vec![42; 4]);
}
