use super::{ktx::Ktx, workload::IndirectBackend};
use fluxel_rhi::api::*;
use std::sync::Arc;

pub(crate) fn buffer<B: IndirectBackend>(
    d: &Device<B>,
    label: &str,
    size: u64,
    usage: BufferUsage,
) -> Result<Buffer<B>> {
    d.create_buffer(&BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    })
}
pub(crate) fn entry(
    binding: u32,
    visibility: Vec<ShaderStage>,
    ty: BindingType,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility,
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}
pub(crate) fn uniform_entry(binding: u32, visibility: Vec<ShaderStage>) -> BindGroupLayoutEntry {
    entry(
        binding,
        visibility,
        BindingType::UniformBuffer {
            dynamic_offset: false,
            min_size: 128,
        },
    )
}
pub(crate) fn layout<B: IndirectBackend>(
    d: &Device<B>,
    dimension: TextureViewDimension,
    label: &str,
) -> Result<BindGroupLayout<B>> {
    d.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some(label.into()),
        entries: vec![
            uniform_entry(0, vec![ShaderStage::Vertex]),
            entry(
                1,
                vec![ShaderStage::Fragment],
                BindingType::SampledTexture {
                    sample: TextureSampleType::Float,
                    dimension,
                    multisampled: false,
                },
            ),
            entry(
                2,
                vec![ShaderStage::Fragment],
                BindingType::Sampler {
                    filtering: true,
                    comparison: false,
                },
            ),
        ],
    })
}
pub(crate) fn uniform_group<B: IndirectBackend>(
    d: &Device<B>,
    l: &BindGroupLayout<B>,
    u: &Buffer<B>,
    label: &str,
) -> Result<BindGroup<B>> {
    d.create_bind_group(&BindGroupDesc {
        label: Some(label.into()),
        layout: l,
        entries: &[BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer {
                buffer: u,
                range: 0..256,
            },
        }],
    })
}
pub(crate) fn textured_group<B: IndirectBackend>(
    d: &Device<B>,
    l: &BindGroupLayout<B>,
    u: &Buffer<B>,
    v: &TextureView<B>,
    s: &Sampler<B>,
    label: &str,
) -> Result<BindGroup<B>> {
    d.create_bind_group(&BindGroupDesc {
        label: Some(label.into()),
        layout: l,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: u,
                    range: 0..256,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(v),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(s),
            },
        ],
    })
}
pub(crate) async fn shader<B: IndirectBackend>(
    d: &Device<B>,
    label: &str,
    bytes: &[u8],
    entry: &str,
    stage: ShaderStage,
) -> Result<Shader<B>> {
    d.shader(ShaderDesc {
        label: Some(label.into()),
        format: ShaderFormat::Dxil,
        bytes: Arc::from(bytes),
        entries: vec![ShaderEntryPoint {
            name: entry.into(),
            stage,
            workgroup_size: None,
        }],
        required_features: vec![],
    })
    .await
}
pub(crate) async fn pipeline<B: IndirectBackend>(
    d: &Device<B>,
    label: &str,
    layout: PipelineLayout<B>,
    format: TextureFormat,
    vs: &[u8],
    vse: &str,
    ps: &[u8],
    pse: &str,
    writes_depth: bool,
    instanced: bool,
) -> Result<RasterPipeline<B>> {
    let mut vertex_buffers = vec![VertexBufferLayout {
        stride: 96,
        step_mode: VertexStepMode::Vertex,
        attributes: vec![
            VertexAttribute {
                location: 0,
                offset: 0,
                format: VertexFormat::Float32(3),
            },
            VertexAttribute {
                location: 1,
                offset: 12,
                format: VertexFormat::Float32(3),
            },
            VertexAttribute {
                location: 2,
                offset: 24,
                format: VertexFormat::Float32(2),
            },
            VertexAttribute {
                location: 3,
                offset: 32,
                format: VertexFormat::Float32(3),
            },
        ],
    }];
    if instanced {
        vertex_buffers.push(VertexBufferLayout {
            stride: 32,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    location: 4,
                    offset: 0,
                    format: VertexFormat::Float32(3),
                },
                VertexAttribute {
                    location: 5,
                    offset: 12,
                    format: VertexFormat::Float32(3),
                },
                VertexAttribute {
                    location: 6,
                    offset: 24,
                    format: VertexFormat::Float32(1),
                },
                VertexAttribute {
                    location: 7,
                    offset: 28,
                    format: VertexFormat::Sint32(1),
                },
            ],
        });
    }
    let face = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    d.raster_pipeline(RasterPipelineDesc {
        label: Some(label.into()),
        layout,
        vertex: PipelineStage {
            module: shader(d, label, vs, vse, ShaderStage::Vertex).await?,
            entry: vse.into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: shader(d, label, ps, pse, ShaderStage::Fragment).await?,
            entry: pse.into(),
            constants: vec![],
        }),
        vertex_buffers,
        color_targets: vec![Some(ColorTargetState {
            format,
            blend: None,
            write_mask: 15,
        })],
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::Back,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: writes_depth,
            depth_compare: CompareFunction::LessEqual,
            stencil_front: face,
            stencil_back: face,
            stencil_read_mask: 0,
            stencil_write_mask: 0,
            depth_bias_constant: 0,
            depth_bias_slope_scale: 0.0,
            depth_bias_clamp: 0.0,
        }),
        multisample: MultisampleState {
            count: 1,
            mask: u64::from(u32::MAX),
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
    .await
}
pub(crate) fn texture<B: IndirectBackend>(
    d: &Device<B>,
    label: &str,
    k: &Ktx,
) -> Result<Texture<B>> {
    let desc = TextureDesc {
        label: Some(label.into()),
        dimension: TextureDimension::D2,
        extent: [k.width, k.height, 1],
        array_layers: k.layers,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: k.mips,
        samples: 1,
        usage: TextureUsage::TEXTURE_BINDING.union(TextureUsage::COPY_DST),
        view_formats: vec![],
    };
    match d.texture_support(&desc) {
        Support::Supported(s) if s.filterable => d.create_texture(&desc),
        Support::Supported(_) => Err(Error::Unsupported(
            "10 needs filterable RGBA8 texture arrays",
        )),
        Support::Unsupported(x) => Err(Error::Unsupported(x)),
    }
}
pub(crate) fn depth<B: IndirectBackend>(
    d: &Device<B>,
    e: [u32; 2],
) -> Result<(Texture<B>, TextureView<B>)> {
    let t = d.create_texture(&TextureDesc {
        label: Some("10 depth".into()),
        dimension: TextureDimension::D2,
        extent: [e[0], e[1], 1],
        array_layers: 1,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let v = d.create_view(
        &t,
        &TextureViewDesc {
            format: TextureFormat::Depth32Float,
            dimension: TextureViewDimension::D2,
            aspects: TextureAspects::DEPTH,
            mips: 0..1,
            layers: 0..1,
        },
    )?;
    Ok((t, v))
}
pub(crate) fn upload_ktx<B: IndirectBackend>(
    e: &mut CommandEncoder<B>,
    t: &Texture<B>,
    k: &Ktx,
) -> Result<()> {
    for (mip, levels) in k.data.iter().enumerate() {
        let w = (k.width >> mip).max(1);
        let h = (k.height >> mip).max(1);
        for (layer, bytes) in levels.iter().enumerate() {
            e.write_texture(
                TextureRegion {
                    texture: t,
                    mip: mip as u32,
                    origin: [0, 0, layer as u32],
                    extent: [w, h, 1],
                    aspect: TextureAspects::COLOR,
                },
                TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: w * 4,
                    rows_per_image: h,
                },
                bytes,
            )?;
        }
    }
    Ok(())
}
pub(crate) fn area(e: [u32; 2]) -> Rect {
    Rect {
        x: 0,
        y: 0,
        width: e[0],
        height: e[1],
    }
}
pub(crate) fn viewport(e: [u32; 2]) -> Viewport {
    Viewport {
        x: 0.0,
        y: 0.0,
        width: e[0] as f32,
        height: e[1] as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}
