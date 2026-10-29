//! Depth format sampling readback across distinct native depth formats.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
fn run_depth_sample_readback(
    format: TextureFormat,
    clear_depth: f32,
    expected: [f32; 4],
    ordinal: u64,
) {
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
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    e.clear_texture(&tex, 0..1, 0..1, TextureClearValue::Depth(clear_depth))
        .unwrap();
    let view = d
        .create_view(
            &tex,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
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
                        sample: TextureSampleType::Depth,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
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
        format: ShaderFormat::Dxil,
        bytes: Arc::from(include_bytes!("assets/depth_sample.dxil").as_slice()),
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
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::DEPTH_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_READ,
                },
            },
            layout: Some((
                ImageLayout::DepthStencilAttachment,
                ImageLayout::ShaderReadOnly,
            )),
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
    assert_eq!(
        sample
            .chunks_exact(4)
            .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
        expected,
        "{format:?}"
    );
}

#[test]
fn depth32_float_sample_readback() {
    run_depth_sample_readback(
        TextureFormat::Depth32Float,
        0.25,
        [0.25, 0.0, 0.0, 1.0],
        99530,
    );
}

#[test]
fn depth16_unorm_sample_readback() {
    run_depth_sample_readback(
        TextureFormat::Depth16Unorm,
        1.0,
        [1.0, 0.0, 0.0, 1.0],
        99531,
    );
}

#[test]
fn depth24_plus_sample_readback() {
    run_depth_sample_readback(
        TextureFormat::Depth24Plus,
        0.25,
        [0.25, 0.0, 0.0, 1.0],
        99532,
    );
}

#[test]
fn depth24_plus_stencil8_sample_readback() {
    run_depth_sample_readback(
        TextureFormat::Depth24PlusStencil8,
        0.25,
        [0.25, 0.0, 0.0, 1.0],
        99533,
    );
}

#[test]
fn depth32_float_stencil8_sample_readback() {
    run_depth_sample_readback(
        TextureFormat::Depth32FloatStencil8,
        0.25,
        [0.25, 0.0, 0.0, 1.0],
        99534,
    );
}
