//! Real GPU coverage for storage descriptor arrays and partially bound holes.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::Dx12Instance;
use std::sync::Arc;

#[path = "support/mod.rs"]
mod support;
use support::block;

#[test]
fn sampled_storage_texture_and_sampler_arrays_are_consumed() {
    run(false, false);
    run(true, false);
    run(false, true);
}
fn run(array: bool, sparse_outputs: bool) {
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
    let texture = |usage, array_layers| {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage,
            view_formats: vec![],
        })
        .unwrap()
    };
    let inputs = [
        texture(
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            1,
        ),
        texture(
            TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING,
            1,
        ),
    ];
    let make_output = || {
        let desc = TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: if array { 2 } else { 1 },
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::STORAGE_BINDING | TextureUsage::COPY_SRC,
            view_formats: vec![],
        };
        if sparse_outputs {
            d.sparse().unwrap().texture(&desc).unwrap()
        } else {
            d.create_texture(&desc).unwrap()
        }
    };
    let outputs = [make_output(), make_output()];
    let view = |t, array| {
        d.create_view(
            t,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: if array {
                    TextureViewDimension::D2Array
                } else {
                    TextureViewDimension::D2
                },
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..if array { 2 } else { 1 },
            },
        )
        .unwrap()
    };
    let ins = [view(&inputs[0], false), view(&inputs[1], false)];
    let outs = [view(&outputs[0], array), view(&outputs[1], array)];
    for sample in [
        TextureSampleType::Sint,
        TextureSampleType::Uint,
        TextureSampleType::Depth,
    ] {
        let invalid_layout = d
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: None,
                entries: vec![BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vec![ShaderStage::Compute],
                    ty: BindingType::SampledTexture {
                        sample,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: BindingCount::One,
                    partially_bound: false,
                    nonuniform_indexing: false,
                }],
            })
            .unwrap();
        assert!(matches!(
            d.create_bind_group(&BindGroupDesc {
                label: None,
                layout: &invalid_layout,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&ins[0])
                }]
            }),
            Err(Error::InvalidState)
        ));
    }
    let wrong_format = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageTexture {
                    format: TextureFormat::Bgra8Unorm,
                    dimension: if array {
                        TextureViewDimension::D2Array
                    } else {
                        TextureViewDimension::D2
                    },
                    access: StorageAccess::Write,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    assert!(matches!(
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &wrong_format,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&outs[0])
            }]
        }),
        Err(Error::InvalidState)
    ));
    let array_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::StorageTexture {
                    format: TextureFormat::Rgba8Unorm,
                    dimension: TextureViewDimension::D2Array,
                    access: StorageAccess::Write,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    assert!(matches!(
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &array_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&ins[0])
            }]
        }),
        Err(Error::InvalidState)
    ));
    let sampler = || {
        d.create_sampler(&SamplerDesc {
            label: None,
            address: [AddressMode::ClampToEdge; 3],
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mip_filter: FilterMode::Nearest,
            lod: 0.0..1.0,
            anisotropy: 1,
            compare: None,
            border: None,
        })
        .unwrap()
    };
    let samplers = [sampler(), sampler()];
    let entry = |binding, ty| BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::Compute],
        ty,
        count: BindingCount::Fixed(2),
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(
                    0,
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
                entry(
                    1,
                    BindingType::Sampler {
                        filtering: false,
                        comparison: false,
                    },
                ),
                entry(
                    2,
                    BindingType::StorageTexture {
                        format: TextureFormat::Rgba8Unorm,
                        dimension: if array {
                            TextureViewDimension::D2Array
                        } else {
                            TextureViewDimension::D2
                        },
                        access: StorageAccess::Write,
                    },
                ),
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
                    resource: BindingResource::Array(
                        ins.iter()
                            .map(|x| Some(BindingResource::TextureView(x)))
                            .collect(),
                    ),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Array(
                        samplers
                            .iter()
                            .map(|x| Some(BindingResource::Sampler(x)))
                            .collect(),
                    ),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Array(
                        outs.iter()
                            .map(|x| Some(BindingResource::TextureView(x)))
                            .collect(),
                    ),
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
        bytes: if array {
            Arc::from(include_bytes!("assets/descriptor_texture_d2array.dxil").as_slice())
        } else {
            Arc::from(include_bytes!("assets/descriptor_texture.dxil").as_slice())
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
    let dst = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 1024,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let empty = TextureUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let render = TextureUses {
        stages: StageMask::COLOR_OUTPUT,
        accesses: AccessMask::COLOR_WRITE,
    };
    let read = TextureUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_READ,
    };
    let write = TextureUses {
        stages: StageMask::COMPUTE,
        accesses: AccessMask::SHADER_WRITE,
    };
    let copy = TextureUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let barrier = |texture, from, to, a, b, layers| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..layers,
            },
            usage: StateTransition { from, to },
            layout: Some((a, b)),
            ownership: None,
        })
    };
    let mut e = d.encoder(lane).unwrap();
    for (i, t) in inputs.iter().enumerate() {
        e.transition_resources(&[barrier(
            t,
            empty,
            render,
            ImageLayout::Undefined,
            ImageLayout::ColorAttachment,
            1,
        )])
        .unwrap();
        e.clear_texture(
            t,
            0..1,
            0..1,
            TextureClearValue::Color(ClearColor::Float(if i == 0 {
                [1., 0., 0., 1.]
            } else {
                [0., 1., 0., 1.]
            })),
        )
        .unwrap();
        e.transition_resources(&[barrier(
            t,
            render,
            read,
            ImageLayout::ColorAttachment,
            ImageLayout::ShaderReadOnly,
            1,
        )])
        .unwrap();
    }
    for t in &outputs {
        e.transition_resources(&[barrier(
            t,
            empty,
            write,
            ImageLayout::Undefined,
            ImageLayout::General,
            if array { 2 } else { 1 },
        )])
        .unwrap();
    }
    {
        let mut p = d.compute().unwrap().begin(&mut e).unwrap();
        p.set_pipeline(&pipeline).unwrap();
        p.set_bind_group(0, &group, &[]).unwrap();
        p.dispatch([1, 1, 1]).unwrap();
        p.finish().unwrap();
    }
    for (i, t) in outputs.iter().enumerate() {
        e.transition_resources(&[barrier(
            t,
            write,
            copy,
            ImageLayout::General,
            ImageLayout::CopySrc,
            if array { 2 } else { 1 },
        )])
        .unwrap();
        e.texture_to_buffer(
            TextureRegion {
                texture: t,
                mip: 0,
                origin: [0, 0, if array { 1 } else { 0 }],
                extent: [1, 1, 1],
                aspect: TextureAspects::COLOR,
            },
            &dst,
            TexelCopyBufferLayout {
                offset: i as u64 * 512,
                bytes_per_row: 256,
                rows_per_image: 1,
            },
        )
        .unwrap();
    }
    let ticket = e.readback_buffer(&dst, 0..1024).unwrap();
    let mut descs = Vec::new();
    let mut binds = Vec::new();
    if sparse_outputs {
        let sparse = d.sparse().unwrap();
        for t in &outputs {
            let tiling = sparse.tiling(t).unwrap();
            let tile = if let Some(tail) = tiling.packed_tail {
                SparseTile::packed_tail(tail)
            } else {
                SparseTile {
                    aspect: None,
                    mip: 0,
                    layer: 0,
                    origin: [0, 0, 0],
                    count: tiling.tile_grid,
                }
            };
            let count =
                u64::from(tile.count[0]) * u64::from(tile.count[1]) * u64::from(tile.count[2]);
            binds.push(SparseBind {
                resource: t.clone(),
                tiles: vec![tile],
                allocation: Some(sparse.allocation(count * tiling.tile_bytes).unwrap()),
                allocation_offset: 0,
            });
        }
        descs.push(StepDesc {
            lane,
            kind: StepKind::SparseBind,
            waits: vec![],
        });
    }
    descs.push(StepDesc {
        lane,
        kind: StepKind::Commands,
        waits: vec![],
    });
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 99003,
            },
            Arc::from(descs),
        )
    };
    let mut steps = Vec::new();
    if sparse_outputs {
        steps.push(BoundStep {
            step: plan.step(0),
            payload: StepPayload::SparseBind(binds),
        });
    }
    steps.push(BoundStep {
        step: plan.step(if sparse_outputs { 1 } else { 0 }),
        payload: StepPayload::Commands(vec![e.finish().unwrap()]),
    });
    // SAFETY: mappings precede all accesses on the same queue.
    let result = unsafe {
        d.queue().submit(SubmitPacket {
            plan,
            steps,
            completion: CompletionRequest::End,
        })
    };
    let point = match result {
        SubmitOutcome::Accepted(r) => r.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("submit rejected"),
    };
    block(d.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(d.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(&bytes[0..4], &[255, 0, 0, 255]);
    assert_eq!(&bytes[512..516], &[0, 255, 0, 255]);
}
