//! Real GPU coverage for storage descriptor arrays and partially bound holes.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block;

#[test]
fn storage_arrays_write_both_descriptors_and_allow_holes() {
    check_storage_arrays(BindingCount::Fixed(2));
    check_storage_arrays(BindingCount::RuntimeSized { max_count: 3 });
}

fn check_storage_arrays(count: BindingCount) {
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
    let make = || {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 4,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let buffers = [make(), make()];
    let make_layout = |partial| {
        d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 4,
                    access: StorageAccess::ReadWrite,
                },
                count,
                partially_bound: partial,
                nonuniform_indexing: false,
            }],
        })
        .unwrap()
    };
    let layout = make_layout(false);
    let resource = |i| BindingResource::StorageBufferView {
        buffer: &buffers[i],
        range: 0..4,
        view: StorageBufferView::Raw,
    };
    let group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Array(vec![Some(resource(0)), Some(resource(1))]),
            }],
        })
        .unwrap();
    let partial = make_layout(true);
    let partial_group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &partial,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Array(vec![None, Some(resource(1))]),
            }],
        })
        .unwrap();
    let empty_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                BindGroupLayoutEntry {
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
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 4,
                        access: StorageAccess::ReadWrite,
                    },
                    count: BindingCount::Fixed(2),
                    partially_bound: true,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 4,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::UniformBuffer {
                        dynamic_offset: false,
                        min_size: 4,
                    },
                    count: BindingCount::Fixed(2),
                    partially_bound: true,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 6,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: BindingCount::Fixed(2),
                    partially_bound: true,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 8,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageTexture {
                        format: TextureFormat::Rgba8Unorm,
                        dimension: TextureViewDimension::D2,
                        access: StorageAccess::Write,
                    },
                    count: BindingCount::Fixed(2),
                    partially_bound: true,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 10,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::Sampler {
                        filtering: false,
                        comparison: false,
                    },
                    count: BindingCount::Fixed(2),
                    partially_bound: true,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 12,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::AccelerationStructure,
                    count: BindingCount::Fixed(2),
                    partially_bound: true,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 14,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 4,
                        access: StorageAccess::Read,
                    },
                    count: BindingCount::Fixed(2),
                    partially_bound: true,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 16,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 4,
                        access: StorageAccess::ReadWrite,
                    },
                    count,
                    partially_bound: true,
                    nonuniform_indexing: false,
                },
            ],
        })
        .unwrap();
    let empty_group = d
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &empty_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: resource(1),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Array(vec![None, None]),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Array(vec![None, None]),
                },
                BindGroupEntry {
                    binding: 6,
                    resource: BindingResource::Array(vec![None, None]),
                },
                BindGroupEntry {
                    binding: 8,
                    resource: BindingResource::Array(vec![None, None]),
                },
                BindGroupEntry {
                    binding: 10,
                    resource: BindingResource::Array(vec![None, None]),
                },
                BindGroupEntry {
                    binding: 12,
                    resource: BindingResource::Array(vec![None, None]),
                },
                BindGroupEntry {
                    binding: 14,
                    resource: BindingResource::Array(vec![None, None]),
                },
                BindGroupEntry {
                    binding: 16,
                    resource: BindingResource::Array(vec![None, None]),
                },
            ],
        })
        .unwrap();
    let empty_pipeline_layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&empty_layout],
            push_constants: &[],
        })
        .unwrap();
    let empty_shader = block(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Dxil,
        bytes: Arc::from(include_bytes!("assets/descriptor_empty.dxil").as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap();
    let empty_pipeline = block(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
        label: None,
        layout: empty_pipeline_layout,
        compute: PipelineStage {
            module: empty_shader,
            entry: "main".into(),
            constants: vec![],
        },
        cache: None,
        required_subgroup_size: None,
    }))
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
        bytes: Arc::from(include_bytes!("assets/descriptor_arrays.dxil").as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap();
    let partial_layout = d
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&partial],
            push_constants: &[],
        })
        .unwrap();
    let partial_shader = block(d.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::Dxil,
        bytes: Arc::from(include_bytes!("assets/descriptor_partial.dxil").as_slice()),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap();
    let partial_pipeline = block(d.compute().unwrap().create_pipeline(ComputePipelineDesc {
        label: None,
        layout: partial_layout,
        compute: PipelineStage {
            module: partial_shader,
            entry: "main".into(),
            constants: vec![],
        },
        cache: None,
        required_subgroup_size: None,
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
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
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
    let read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |b, from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: b,
                range: 0..4,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
    let mut e = d.encoder(lane).unwrap();
    e.transition_resources(&[
        barrier(&buffers[0], empty, write),
        barrier(&buffers[1], empty, write),
    ])
    .unwrap();
    {
        let mut p = d.compute().unwrap().begin(&mut e).unwrap();
        p.set_pipeline(&pipeline).unwrap();
        p.set_bind_group(0, &group, &[]).unwrap();
        p.dispatch([1, 1, 1]).unwrap();
        p.finish().unwrap();
    }
    e.transition_resources(&[
        barrier(&buffers[0], write, read),
        barrier(&buffers[1], write, read),
    ])
    .unwrap();
    e.copy_buffer(&buffers[0], 0, &dst, 0, 4).unwrap();
    e.copy_buffer(&buffers[1], 0, &dst, 4, 4).unwrap();
    e.transition_resources(&[barrier(&buffers[1], read, write)])
        .unwrap();
    {
        let mut p = d.compute().unwrap().begin(&mut e).unwrap();
        p.set_pipeline(&partial_pipeline).unwrap();
        p.set_bind_group(0, &partial_group, &[]).unwrap();
        p.dispatch([1, 1, 1]).unwrap();
        p.finish().unwrap();
    }
    e.transition_resources(&[barrier(&buffers[1], write, read)])
        .unwrap();
    e.copy_buffer(&buffers[1], 0, &dst, 8, 4).unwrap();
    e.transition_resources(&[barrier(&buffers[1], read, write)])
        .unwrap();
    {
        let mut p = d.compute().unwrap().begin(&mut e).unwrap();
        p.set_pipeline(&empty_pipeline).unwrap();
        p.set_bind_group(0, &empty_group, &[]).unwrap();
        p.dispatch([1, 1, 1]).unwrap();
        p.finish().unwrap();
    }
    e.transition_resources(&[barrier(&buffers[1], write, read)])
        .unwrap();
    e.copy_buffer(&buffers[1], 0, &dst, 12, 4).unwrap();
    let ticket = e.readback_buffer(&dst, 0..16).unwrap();
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 99001,
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
    assert_eq!(
        &*bytes,
        &[41, 0, 0, 0, 73, 0, 0, 0, 99, 0, 0, 0, 123, 0, 0, 0]
    );
}
