//! Copy complete depth/stencil planes between textures and sample the result.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
#[test]
fn depth_stencil_plane_copy_readback() {
    for (index, format) in [
        TextureFormat::Depth24PlusStencil8,
        TextureFormat::Depth32FloatStencil8,
    ]
    .into_iter()
    .enumerate()
    {
        check(format, TextureAspects::DEPTH, index as u64 * 2);
        check(format, TextureAspects::STENCIL, index as u64 * 2 + 1);
    }
}
fn check(format: TextureFormat, aspect: TextureAspects, index: u64) {
    let expected = if aspect == TextureAspects::DEPTH {
        [0.25f32.to_bits(), 0, 0, 1.0f32.to_bits()]
    } else {
        [37, 0, 0, 0]
    };
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
            usage: TextureUsage::RENDER_ATTACHMENT
                | TextureUsage::TEXTURE_BINDING
                | TextureUsage::COPY_SRC
                | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let dst = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 1,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap();
    let mut e = d.encoder(lane).unwrap();
    let attachment_view = d
        .create_view(
            &tex,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture: &tex,
            aspects: TextureAspects::DEPTH | TextureAspects::STENCIL,
            mips: 0..1,
            layers: 0..1,
        },
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to: TextureUses {
                stages: StageMask::FRAGMENT,
                accesses: AccessMask::DEPTH_WRITE,
            },
        },
        layout: Some((ImageLayout::Undefined, ImageLayout::DepthStencilAttachment)),
        ownership: None,
    })])
    .unwrap();
    {
        let pass = e
            .begin_raster(&RasterPassDesc {
                label: None,
                colors: &[],
                depth_stencil: Some(DepthStencilAttachment {
                    view: AttachmentView::Texture(&attachment_view),
                    depth: Some((LoadOp::Clear(0.25), StoreOp::Store)),
                    stencil: Some((LoadOp::Clear(37), StoreOp::Store)),
                    depth_read_only: false,
                    stencil_read_only: false,
                }),
                render_area: Rect {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 8,
                },
                multiview_mask: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                secondary_inheritance: None,
            })
            .unwrap();
        pass.finish().unwrap();
    }
    e.transition_resources(&[
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: aspect,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::FRAGMENT,
                    accesses: AccessMask::DEPTH_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::DepthStencilAttachment, ImageLayout::CopySrc)),
            ownership: None,
        }),
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &dst,
                aspects: aspect,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
            },
            layout: Some((ImageLayout::Undefined, ImageLayout::CopyDst)),
            ownership: None,
        }),
    ])
    .unwrap();
    e.copy_texture(
        TextureRegion {
            texture: &tex,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 1],
            aspect,
        },
        TextureRegion {
            texture: &dst,
            mip: 0,
            origin: [0, 0, 0],
            extent: [8, 8, 1],
            aspect,
        },
    )
    .unwrap();
    let view = d
        .create_view(
            &dst,
            &TextureViewDesc {
                format,
                dimension: TextureViewDimension::D2,
                aspects: aspect,
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
                        sample: if aspect == TextureAspects::DEPTH {
                            TextureSampleType::Depth
                        } else {
                            TextureSampleType::Uint
                        },
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
        bytes: Arc::from(if aspect == TextureAspects::DEPTH {
            include_bytes!("assets/depth_sample.dxil").as_slice()
        } else {
            include_bytes!("assets/stencil_plane_sample.dxil").as_slice()
        }),
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
                texture: &dst,
                aspects: aspect,
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
            layout: Some((ImageLayout::CopyDst, ImageLayout::ShaderReadOnly)),
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
                ordinal: 99530 + index as u64,
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
            .map(|v| u32::from_ne_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
        expected,
        "{format:?}"
    );
}
