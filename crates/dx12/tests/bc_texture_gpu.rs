//! BC texture uploads and readback preserve encoded blocks.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
#[test]
fn bc_color_formats_sample_and_readback() {
    let red = [0x00, 0xf8, 0x00, 0xf8, 0, 0, 0, 0];
    let alpha = [0xff, 0xff, 0, 0, 0, 0, 0, 0];
    for (index, (format, encoded, expected)) in [
        (
            TextureFormat::Bc1RgbaUnorm,
            red.to_vec(),
            [1.0, 0.0, 0.0, 1.0],
        ),
        (
            TextureFormat::Bc2RgbaUnorm,
            [vec![0xff; 8], red.to_vec()].concat(),
            [1.0, 0.0, 0.0, 1.0],
        ),
        (
            TextureFormat::Bc3RgbaUnorm,
            [alpha.to_vec(), red.to_vec()].concat(),
            [1.0, 0.0, 0.0, 1.0],
        ),
        (
            TextureFormat::Bc4RUnorm,
            alpha.to_vec(),
            [1.0, 0.0, 0.0, 1.0],
        ),
        (
            TextureFormat::Bc5RgUnorm,
            [alpha.to_vec(), alpha.to_vec()].concat(),
            [1.0, 1.0, 0.0, 1.0],
        ),
    ]
    .into_iter()
    .enumerate()
    {
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
                array_layers: 2,
                format,
                mip_levels: 1,
                samples: 1,
                usage: TextureUsage::COPY_SRC
                    | TextureUsage::COPY_DST
                    | TextureUsage::TEXTURE_BINDING,
                view_formats: vec![],
            })
            .unwrap();
        let dst = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 1024,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        let data = encoded.repeat(4);
        let row = encoded.len() * 2;
        let region = || TextureRegion {
            texture: &tex,
            mip: 0,
            origin: [0, 0, 1],
            extent: [8, 8, 1],
            aspect: TextureAspects::COLOR,
        };
        let mut e = d.encoder(lane).unwrap();
        e.write_texture(
            region(),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: row as u32,
                rows_per_image: 2,
            },
            &data,
        )
        .unwrap();
        let view = d
            .create_view(
                &tex,
                &TextureViewDesc {
                    format,
                    dimension: TextureViewDimension::D2Array,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 1..2,
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
                            dimension: TextureViewDimension::D2Array,
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
            bytes: Arc::from(include_bytes!("assets/bc_texture/sample.dxil").as_slice()),
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
                    layers: 1..2,
                },
                usage: StateTransition {
                    from: TextureUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    to: TextureUses {
                        stages: StageMask::COMPUTE,
                        accesses: AccessMask::SHADER_READ,
                    },
                },
                layout: Some((ImageLayout::Common, ImageLayout::ShaderReadOnly)),
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
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..2,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COMPUTE,
                    accesses: AccessMask::SHADER_READ,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::ShaderReadOnly, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
        e.texture_to_buffer(
            region(),
            &dst,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
        )
        .unwrap();
        let ticket = e.readback_buffer(&dst, 0..1024).unwrap();
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal: 99801 + index as u64,
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
        let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
        let sample = unsafe { block(d.read_after(sample_ticket, &point)).unwrap().bytes };
        assert_eq!(
            sample
                .chunks_exact(4)
                .map(|v| f32::from_ne_bytes(v.try_into().unwrap()))
                .collect::<Vec<_>>(),
            expected,
            "{format:?}"
        );
        for y in 0..2 {
            assert_eq!(
                &bytes[y * 256..y * 256 + row],
                &data[y * row..y * row + row]
            );
        }
    }
}

#[test]
fn bc_formats_upload_and_readback() {
    use TextureFormat::*;
    let formats = [
        (Bc1RgbaUnorm, 8),
        (Bc1RgbaUnormSrgb, 8),
        (Bc2RgbaUnorm, 16),
        (Bc2RgbaUnormSrgb, 16),
        (Bc3RgbaUnorm, 16),
        (Bc3RgbaUnormSrgb, 16),
        (Bc4RUnorm, 8),
        (Bc4RSnorm, 8),
        (Bc5RgUnorm, 16),
        (Bc5RgSnorm, 16),
        (Bc6hRgbUfloat, 16),
        (Bc6hRgbFloat, 16),
        (Bc7RgbaUnorm, 16),
        (Bc7RgbaUnormSrgb, 16),
    ];
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|v| v.class == LaneClass::Graphics)
        .unwrap()
        .id;
    for (index, (format, block_size)) in formats.into_iter().enumerate() {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [8, 8, 1],
            array_layers: 2,
            format,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST | TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        };
        if matches!(d.texture_support(&desc), Support::Unsupported(_)) {
            continue;
        }
        let tex = d.create_texture(&desc).unwrap();
        let dst = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 512,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        let row = 2 * block_size;
        let data: Vec<_> = (0..row * 2)
            .map(|n| (n as u8).wrapping_mul(7).wrapping_add(3))
            .collect();
        let region = || TextureRegion {
            texture: &tex,
            mip: 0,
            origin: [0, 0, 1],
            extent: [8, 8, 1],
            aspect: TextureAspects::COLOR,
        };
        let mut e = d.encoder(lane).unwrap();
        e.write_texture(
            region(),
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: row as u32,
                rows_per_image: 2,
            },
            &data,
        )
        .unwrap();
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..2,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::Common, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
        e.texture_to_buffer(
            region(),
            &dst,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
        )
        .unwrap();
        let ticket = e.readback_buffer(&dst, 0..512).unwrap();
        let source = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 512,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
                memory: MemoryClass::Device,
            })
            .unwrap();
        let mut staged = vec![0u8; 512];
        for y in 0..2 {
            staged[y * 256..y * 256 + row].copy_from_slice(&data[y * row..y * row + row]);
        }
        e.write_buffer(&source, 0, &staged).unwrap();
        e.transition_resources(&[
            ResourceTransition::Buffer(BufferBarrier {
                target: BufferSlice {
                    buffer: &source,
                    range: 0..512,
                },
                usage: StateTransition {
                    from: BufferUses {
                        stages: StageMask::EMPTY,
                        accesses: AccessMask::EMPTY,
                    },
                    to: BufferUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                },
                ownership: None,
            }),
            ResourceTransition::Texture(TextureBarrier {
                target: Subresources {
                    texture: &tex,
                    aspects: TextureAspects::COLOR,
                    mips: 0..1,
                    layers: 1..2,
                },
                usage: StateTransition {
                    from: TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_READ,
                    },
                    to: TextureUses {
                        stages: StageMask::COPY,
                        accesses: AccessMask::COPY_WRITE,
                    },
                },
                layout: Some((ImageLayout::CopySrc, ImageLayout::CopyDst)),
                ownership: None,
            }),
        ])
        .unwrap();
        e.buffer_to_texture(
            &source,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
            region(),
        )
        .unwrap();
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &tex,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..2,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::CopyDst, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
        let dst2 = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 512,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        e.texture_to_buffer(
            region(),
            &dst2,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
        )
        .unwrap();
        let ticket2 = e.readback_buffer(&dst2, 0..512).unwrap();
        let copied = d.create_texture(&desc).unwrap();
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &copied,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..2,
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
            layout: Some((ImageLayout::Common, ImageLayout::CopyDst)),
            ownership: None,
        })])
        .unwrap();
        let copied_region = || TextureRegion {
            texture: &copied,
            mip: 0,
            origin: [0, 0, 1],
            extent: [8, 8, 1],
            aspect: TextureAspects::COLOR,
        };
        e.copy_texture(region(), copied_region()).unwrap();
        e.transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &copied,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 1..2,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_WRITE,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::CopyDst, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
        let dst3 = d
            .create_buffer(&BufferDesc {
                label: None,
                size: 512,
                usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
                memory: MemoryClass::Readback,
            })
            .unwrap();
        e.texture_to_buffer(
            copied_region(),
            &dst3,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
        )
        .unwrap();
        let ticket3 = e.readback_buffer(&dst3, 0..512).unwrap();
        let plan = unsafe {
            SubmissionPlan::from_compiled(
                PlanId {
                    device: d.id(),
                    ordinal: 99810 + index as u64,
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
            _ => panic!("submit {format:?}"),
        };
        block(d.wait_completion(&point)).unwrap();
        let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
        let bytes2 = unsafe { block(d.read_after(ticket2, &point)).unwrap().bytes };
        let bytes3 = unsafe { block(d.read_after(ticket3, &point)).unwrap().bytes };
        for y in 0..2 {
            assert_eq!(
                &bytes[y * 256..y * 256 + row],
                &data[y * row..y * row + row],
                "{format:?}"
            );
            assert_eq!(
                &bytes2[y * 256..y * 256 + row],
                &data[y * row..y * row + row],
                "buffer copy {format:?}"
            );
            assert_eq!(
                &bytes3[y * 256..y * 256 + row],
                &data[y * row..y * row + row],
                "texture copy {format:?}"
            );
        }
    }
}
