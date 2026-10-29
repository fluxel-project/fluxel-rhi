//! Real GPU check that bind groups survive switching sampler pipeline layouts.
#![cfg(all(windows, feature = "hardware-tests"))]
#[path = "../support/mod.rs"]
mod support;
use std::sync::Arc;
use support::block;
#[cfg(feature = "hardware-tests")]
#[test]
fn sampler_layout_switch_retains_bind_group() {
    use fluxel_rhi_core::api::*;
    use fluxel_rhi_vulkan::VulkanInstance;
    let desc = ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from(&include_bytes!("../assets/sampler/clamp_mixed_switch.spv")[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    };
    let device = block(VulkanInstance::new().unwrap().open_default()).unwrap();
    let lane = device
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let sampler = device
        .create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToZero; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })
        .unwrap();
    let texture = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [2, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let texture_view = device
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let integer = device
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Uint,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let integer_view = device
        .create_view(
            &integer,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Uint,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let data = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let readback = device
        .create_buffer(&BufferDesc {
            label: None,
            size: 32,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let group_layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::Sampler {
                        filtering: false,
                        comparison: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::SampledTexture {
                        sample: TextureSampleType::Uint,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::StorageBuffer {
                        dynamic_offset: false,
                        min_size: 32,
                        access: StorageAccess::ReadWrite,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                },
            ],
        })
        .unwrap();
    let group = device
        .create_bind_group(&BindGroupDesc {
            label: None,
            layout: &group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Sampler(&sampler),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&integer_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&texture_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::StorageBufferView {
                        buffer: &data,
                        range: 0..32,
                        view: StorageBufferView::Raw,
                    },
                },
            ],
        })
        .unwrap();
    let layout = device
        .create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&group_layout],
            push_constants: &[],
        })
        .unwrap();
    let shader = block(device.shader(desc)).unwrap();
    let pipeline = block(
        device
            .compute()
            .unwrap()
            .create_pipeline(ComputePipelineDesc {
                label: None,
                layout: layout.clone(),
                compute: PipelineStage {
                    module: shader,
                    entry: "main".into(),
                    constants: vec![],
                },
                cache: None,
                required_subgroup_size: None,
            }),
    )
    .unwrap();
    let plain_shader = block(device.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from(&include_bytes!("../assets/sampler/clamp_float.spv")[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    }))
    .unwrap();
    let plain = block(
        device
            .compute()
            .unwrap()
            .create_pipeline(ComputePipelineDesc {
                label: None,
                layout: layout.clone(),
                compute: PipelineStage {
                    module: plain_shader,
                    entry: "main".into(),
                    constants: vec![],
                },
                cache: None,
                required_subgroup_size: None,
            }),
    )
    .unwrap();
    let integer_shader = block(device.shader(ShaderDesc {
        label: None,
        format: ShaderFormat::SpirV,
        bytes: Arc::from(&include_bytes!("../assets/sampler/clamp_integer.spv")[..]),
        entries: vec![ShaderEntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            workgroup_size: Some([1, 1, 1]),
        }],
        required_features: vec![],
    }))
    .unwrap();
    let integer_pipeline = block(
        device
            .compute()
            .unwrap()
            .create_pipeline(ComputePipelineDesc {
                label: None,
                layout,
                compute: PipelineStage {
                    module: integer_shader,
                    entry: "main".into(),
                    constants: vec![],
                },
                cache: None,
                required_subgroup_size: None,
            }),
    )
    .unwrap();
    let mut encoder = device.encoder(lane).unwrap();
    encoder
        .write_texture(
            TextureRegion {
                texture: &texture,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 8,
                rows_per_image: 1,
            },
            &[10, 0, 0, 255, 200, 0, 0, 255],
        )
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &texture,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_READ,
                },
            },
            layout: Some((ImageLayout::Common, ImageLayout::ShaderReadOnly)),
            ownership: None,
        })])
        .unwrap();
    encoder
        .write_texture(
            TextureRegion {
                texture: &integer,
                mip: 0,
                origin: [0, 0, 0],
                extent: [1, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 4,
                rows_per_image: 1,
            },
            &[73, 0, 0, 255],
        )
        .unwrap();
    encoder
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &integer,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_READ,
                },
            },
            layout: Some((ImageLayout::Common, ImageLayout::ShaderReadOnly)),
            ownership: None,
        })])
        .unwrap();
    let transition = |from, to| {
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &data,
                range: 0..32,
            },
            usage: StateTransition { from, to },
            ownership: None,
        })
    };
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
    encoder
        .transition_resources(&[transition(empty, write)])
        .unwrap();
    let mut pass = device.compute().unwrap().begin(&mut encoder).unwrap();
    pass.set_pipeline(&plain).unwrap();
    pass.set_bind_group(0, &group, &[]).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.set_pipeline(&pipeline).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.set_pipeline(&integer_pipeline).unwrap();
    pass.dispatch([1, 1, 1]).unwrap();
    pass.finish().unwrap();
    encoder
        .transition_resources(&[transition(write, read)])
        .unwrap();
    encoder.copy_buffer(&data, 0, &readback, 0, 32).unwrap();
    let ticket = encoder.readback_buffer(&readback, 0..32).unwrap();
    let work = encoder.finish().unwrap();
    // SAFETY: the plan and recorded work use the same device and lane.
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal: 19991,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    // SAFETY: resources and work are retained until completion below.
    let result = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    let point = match result {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submission rejected"),
    };
    block(device.wait_completion(&point)).unwrap();
    // SAFETY: the completion point covers the readback ticket.
    let bytes = unsafe { block(device.read_after(ticket, &point)).unwrap().bytes };
    let values: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(values[0], 0);
    assert_eq!(values[1], 73);
    assert_eq!(values[4], 0);
    assert_eq!(values[5], 73);
    assert_eq!(f32::from_bits(values[2]), 0.0);
    assert!((f32::from_bits(values[3]) - 200.0 / 255.0).abs() < 0.001);
}
