//! Browser checks that rejected bind groups do not poison later GPU work.

use super::*;

#[wasm_bindgen_test(async)]
async fn invalid_binding_is_rejected_before_a_later_gpu_submit() {
    let d = open().await;
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Compute],
                ty: BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 16,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let copy_only = buffer(
        &d,
        16,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    assert!(
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &copy_only,
                    range: 0..16,
                },
            }],
        })
        .is_err()
    );

    let storage_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
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
            }],
        })
        .unwrap();
    let storage = buffer(
        &d,
        16,
        BufferUsage::STORAGE | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    assert!(
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &storage_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &storage,
                    range: 0..6,
                },
            }],
        })
        .is_err()
    );
    let uniform = buffer(
        &d,
        16,
        BufferUsage::UNIFORM | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    assert!(
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &uniform,
                    range: 0..4,
                },
            }],
        })
        .is_err()
    );

    let sampler_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::Sampler {
                    filtering: true,
                    comparison: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    let nearest = d
        .create_sampler(&SamplerDesc {
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
        .unwrap();
    d.create_bind_group(&BindGroupDesc {
        label: None,
        layout: &sampler_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Sampler(&nearest),
        }],
    })
    .unwrap();
    let stencil = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format: TextureFormat::Stencil8,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let stencil_view = d
        .create_view(
            &stencil,
            &TextureViewDesc {
                format: TextureFormat::Stencil8,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::STENCIL,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let stencil_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::Uint,
                    dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    d.create_bind_group(&BindGroupDesc {
        label: None,
        layout: &stencil_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&stencil_view),
        }],
    })
    .unwrap();
    let stencil_float_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::UnfilterableFloat,
                    dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    assert!(
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &stencil_float_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&stencil_view),
            }],
        })
        .is_err()
    );

    let float = d
        .create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1, 1, 1],
            array_layers: 1,
            format: TextureFormat::R32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::TEXTURE_BINDING,
            view_formats: vec![],
        })
        .unwrap();
    let float_view = d
        .create_view(
            &float,
            &TextureViewDesc {
                format: TextureFormat::R32Float,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let unfilterable_layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::UnfilterableFloat,
                    dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    d.create_bind_group(&BindGroupDesc {
        label: None,
        layout: &unfilterable_layout,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::TextureView(&float_view),
        }],
    })
    .unwrap();

    // The local rejection above must not become a deferred browser validation
    // error. A following real queue submission and readback still completes.
    let output = buffer(
        &d,
        4,
        BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
        MemoryClass::Device,
    );
    let mut encoder = d.encoder(d.lanes()[0].id).unwrap();
    encoder
        .write_buffer(&output, 0, &0xdecafbad_u32.to_le_bytes())
        .unwrap();
    let ticket = encoder.readback_buffer(&output, 0..4).unwrap();
    let point = submit(&d, encoder).await;
    // SAFETY: the point follows this fixture's write and readback commands.
    assert_eq!(
        unsafe { d.read_after(ticket, &point).await }.unwrap().bytes,
        0xdecafbad_u32.to_le_bytes(),
    );
}
