//! Multisample texture sampling across D2, D2Array, and WGSL routes.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
fn run_multisample_view_readback(
    view_dimension: TextureViewDimension,
    array_layers: u32,
    shader_format: ShaderFormat,
    shader_bytes: Arc<[u8]>,
    ordinal: u64,
) {
    let format = TextureFormat::Rgba8Unorm;

    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|v| v.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let tex = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers,
            format,
            mip_levels: 1,
            samples: 4,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &tex,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..array_layers,
        },
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to: TextureUses {
                stages: StageMask::COLOR_OUTPUT,
                accesses: AccessMask::COLOR_WRITE,
            },
        },
        layout: Some((ImageLayout::Common, ImageLayout::ColorAttachment)),
        ownership: None,
    })])
    .unwrap();
    e.clear_texture(
        &tex,
        0..1,
        0..array_layers,
        TextureClearValue::Color(ClearColor::Float([0.5, 0.0, 0.0, 1.0])),
    )
    .unwrap();
    let view = d
        .create_view(
            &tex,
            &TextureViewDesc {
                format,
                dimension: view_dimension,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..array_layers,
            },
        )
        .unwrap();
    let output = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::STORAGE | BufferUsage::COPY_SRC,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let output_read = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: view_dimension,
                        multisampled: true,
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
                        min_size: 16,
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
                    resource: BindingResource::TextureView(&view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::StorageBufferView {
                        buffer: &output,
                        range: 0..16,
                        view: StorageBufferView::Raw,
                    },
                },
            ],
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
        format: shader_format,
        bytes: shader_bytes,
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
        required_subgroup_size: None,
        cache: None,
    }))
    .unwrap();
    e.transition_resources(&[
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..array_layers,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COLOR_OUTPUT,
                    accesses: AccessMask::COLOR_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_READ,
                },
            },
            layout: Some((ImageLayout::ColorAttachment, ImageLayout::ShaderReadOnly)),
            ownership: None,
        }),
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &output,
                range: 0..16,
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
        }),
    ])
    .unwrap();
    {
        let mut pass = d.compute().unwrap().begin(&mut e).unwrap();
        pass.set_pipeline(&pipeline).unwrap();
        pass.set_bind_group(0, &group, &[]).unwrap();
        pass.dispatch([1, 1, 1]).unwrap();
        pass.finish().unwrap();
    }
    e.transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer: &output,
            range: 0..16,
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
    e.copy_buffer(&output, 0, &output_read, 0, 16).unwrap();
    let sample_ticket = e.readback_buffer(&output_read, 0..16).unwrap();
    let plan = unsafe {
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
    let step = plan.step(0);
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
        _ => panic!("submission rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let sample = unsafe { block(d.read_after(sample_ticket, &point)).unwrap().bytes };
    let value = f32::from_ne_bytes(sample[0..4].try_into().unwrap());
    assert!((value - 0.50196).abs() < 0.003, "multisample value {value}");
}

#[test]
fn multisample_2d_dxil_view_readback() {
    run_multisample_view_readback(
        TextureViewDimension::D2,
        1,
        ShaderFormat::Dxil,
        Arc::from(include_bytes!("assets/msaa_sample.dxil").as_slice()),
        99530,
    );
}

#[test]
fn multisample_array_single_layer_dxil_readback() {
    run_multisample_view_readback(
        TextureViewDimension::D2Array,
        1,
        ShaderFormat::Dxil,
        Arc::from(include_bytes!("assets/msaa_array_single.dxil").as_slice()),
        99531,
    );
}

#[test]
fn multisample_array_layers_dxil_readback() {
    run_multisample_view_readback(
        TextureViewDimension::D2Array,
        2,
        ShaderFormat::Dxil,
        Arc::from(include_bytes!("assets/msaa_array_sample.dxil").as_slice()),
        99532,
    );
}

#[test]
fn multisample_2d_wgsl_readback() {
    run_multisample_view_readback(
        TextureViewDimension::D2,
        1,
        ShaderFormat::Wgsl,
        Arc::from(&b"@group(0) @binding(0) var image: texture_multisampled_2d<f32>; @group(0) @binding(1) var<storage, read_write> result: array<vec4<u32>>; @compute @workgroup_size(1) fn main() { result[0] = bitcast<vec4<u32>>(textureLoad(image, vec2<i32>(1, 1), 1)); }"[..]),
        99533,
    );
}
