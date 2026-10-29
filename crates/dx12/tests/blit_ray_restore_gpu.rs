//! DXR dynamic CBV/SRV/UAV offsets and descriptor lifetime.
#![cfg(all(windows, feature = "hardware-tests"))]
use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{Dx12, Dx12Instance};
use std::sync::Arc;
#[path = "support/mod.rs"]
mod support;
use support::block;
fn barrier(
    buffer: &Buffer<Dx12>,
    from: BufferUses,
    to: BufferUses,
) -> ResourceTransition<'_, Dx12> {
    ResourceTransition::Buffer(BufferBarrier {
        target: BufferSlice {
            buffer,
            range: 0..64,
        },
        usage: StateTransition { from, to },
        ownership: None,
    })
}
#[test]
fn ray_bindings_survive_compute_blit() {
    let d = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    if !d.capabilities().ray {
        eprintln!("SKIP: DXR pipeline unsupported");
        return;
    }
    let ray = d.ray().unwrap();
    let lane = d
        .lanes()
        .iter()
        .find(|l| l.class == LaneClass::Compute)
        .unwrap()
        .id;
    let buffer = |usage| {
        d.create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage,
            memory: MemoryClass::Device,
        })
        .unwrap()
    };
    let uniform = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 512,
            usage: BufferUsage::UNIFORM | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let input = buffer(BufferUsage::STORAGE | BufferUsage::COPY_DST);
    let output = buffer(BufferUsage::STORAGE | BufferUsage::COPY_SRC);
    let readback = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 16,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let entry = |binding, access| BindGroupLayoutEntry {
        binding,
        visibility: vec![ShaderStage::RayGeneration],
        ty: BindingType::StorageBuffer {
            dynamic_offset: true,
            min_size: 16,
            access,
        },
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    };
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(0, StorageAccess::Read),
                entry(1, StorageAccess::ReadWrite),
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: vec![ShaderStage::RayGeneration],
                    ty: BindingType::UniformBuffer {
                        dynamic_offset: true,
                        min_size: 16,
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
                    resource: BindingResource::StorageBufferView {
                        buffer: &input,
                        range: 0..16,
                        view: StorageBufferView::Raw,
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::StorageBufferView {
                        buffer: &output,
                        range: 0..16,
                        view: StorageBufferView::Raw,
                    },
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Buffer {
                        buffer: &uniform,
                        range: 0..256,
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
        bytes: Arc::from(&include_bytes!("assets/ray_dynamic.dxil")[..]),
        entries: vec![ShaderEntryPoint {
            name: "raygen".into(),
            stage: ShaderStage::RayGeneration,
            workgroup_size: None,
        }],
        required_features: vec![],
    }))
    .unwrap();
    let pipeline = block(ray.create_pipeline(RayPipelineDesc {
        layout: pl,
        shaders: vec![PipelineStage {
            module: shader,
            entry: "raygen".into(),
            constants: vec![],
        }],
        groups: vec![RayGroup {
            general: Some(0),
            closest_hit: None,
            any_hit: None,
            intersection: None,
            local_constants: None,
        }],
        max_recursion: 1,
        payload_bytes: 4,
        hit_attribute_bytes: 8,
        cache: None,
    }))
    .unwrap();
    for offsets in [
        &[1, 16, 256][..],
        &[52, 16, 256][..],
        &[16, 52, 256][..],
        &[16, 16, 1][..],
        &[16, 16, 512][..],
        &[16, 16][..],
    ] {
        let mut e = d.encoder(lane).unwrap();
        ray.set_pipeline(&mut e, &pipeline).unwrap();
        assert!(
            ray.set_bind_group(&mut e, 0, &group, offsets).is_err(),
            "accepted {offsets:?}"
        );
    }
    let props = ray.shader_table_properties().unwrap();
    let stride = u64::from(props.identifier_size).div_ceil(u64::from(props.record_alignment))
        * u64::from(props.record_alignment);
    let table_buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 64,
            usage: BufferUsage::SHADER_TABLE | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let table = ShaderTable {
        raygen: ShaderTableRegion {
            buffer: table_buffer.clone(),
            offset: 0,
            size: stride,
            stride,
        },
        miss: None,
        hit: None,
        callable: None,
    };
    let identifier = ray.shader_group_identifier(&pipeline, 0).unwrap();
    let mut e = d.encoder(lane).unwrap();
    let mut data = [0u8; 64];
    data[16..20].copy_from_slice(&42u32.to_le_bytes());
    e.write_buffer(&input, 0, &data).unwrap();
    let mut uniforms = [0u8; 512];
    uniforms[256..260].copy_from_slice(&99u32.to_le_bytes());
    e.write_buffer(&uniform, 0, &uniforms).unwrap();
    e.write_buffer(&table_buffer, 0, &identifier).unwrap();
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let read = BufferUses {
        stages: StageMask::RAY,
        accesses: AccessMask::SHADER_READ,
    };
    let write = BufferUses {
        stages: StageMask::RAY,
        accesses: AccessMask::SHADER_WRITE,
    };
    let copy = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    e.transition_resources(&[
        barrier(&input, empty, read),
        barrier(&output, empty, write),
        barrier(
            &table_buffer,
            empty,
            BufferUses {
                stages: StageMask::RAY,
                accesses: AccessMask::RAY_TRACING_SHADER_DATA_READ,
            },
        ),
        ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &uniform,
                range: 0..512,
            },
            usage: StateTransition {
                from: empty,
                to: read,
            },
            ownership: None,
        }),
    ])
    .unwrap();
    ray.set_pipeline(&mut e, &pipeline).unwrap();
    ray.set_bind_group(&mut e, 0, &group, &[16, 16, 256])
        .unwrap();
    ray.trace(&mut e, &pipeline, &table, [1, 1, 1]).unwrap();
    e.transition_resources(&[barrier(&input, read, empty)])
        .unwrap();
    e.write_buffer(&input, 16, &23u32.to_le_bytes()).unwrap();
    e.transition_resources(&[barrier(&input, empty, read)])
        .unwrap();
    let texture = || {
        d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [4, 4, 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            view_formats: vec![],
        })
        .unwrap()
    };
    let source = texture();
    let target = texture();
    let tbarrier = |texture, to, layout| {
        ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to,
            },
            layout: Some((ImageLayout::Common, layout)),
            ownership: None,
        })
    };
    e.transition_resources(&[
        tbarrier(
            &source,
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
            ImageLayout::CopySrc,
        ),
        tbarrier(
            &target,
            TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_WRITE,
            },
            ImageLayout::CopyDst,
        ),
    ])
    .unwrap();
    let region = |texture| TextureRegion {
        texture,
        mip: 0,
        origin: [0, 0, 0],
        extent: [4, 4, 1],
        aspect: TextureAspects::COLOR,
    };
    e.blit(region(&source), region(&target), FilterMode::Nearest)
        .unwrap();
    ray.trace(&mut e, &pipeline, &table, [1, 1, 1]).unwrap();
    e.transition_resources(&[barrier(&output, write, copy)])
        .unwrap();
    e.copy_buffer(&output, 16, &readback, 0, 16).unwrap();
    let ticket = e.readback_buffer(&readback, 0..16).unwrap();
    let work = e.finish().unwrap();
    drop(group);
    drop(input);
    drop(output);
    drop(pipeline);
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: d.id(),
                ordinal: 97134,
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
    let result: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(&result[..2], &[23, 99]);
}
