//! Offscreen scene resource, shader, pipeline, and transform helpers.
use super::super::*;

pub(crate) fn buf<B: OffscreenBackend>(
    d: &Device<B>,
    n: &str,
    size: u64,
    usage: BufferUsage,
) -> Result<Buffer<B>> {
    d.create_buffer(&BufferDesc {
        label: Some(n.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    })
}
pub(crate) fn entry(b: u32, v: Vec<ShaderStage>, ty: BindingType) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding: b,
        visibility: v,
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}
pub(crate) fn uniform_entry(b: u32, v: Vec<ShaderStage>) -> BindGroupLayoutEntry {
    entry(
        b,
        v,
        BindingType::UniformBuffer {
            dynamic_offset: false,
            min_size: 64,
        },
    )
}
pub(crate) fn ug<B: OffscreenBackend>(
    d: &Device<B>,
    l: &BindGroupLayout<B>,
    b: &Buffer<B>,
    n: &str,
) -> Result<BindGroup<B>> {
    let e = [BindGroupEntry {
        binding: 0,
        resource: BindingResource::Buffer {
            buffer: b,
            range: 0..256,
        },
    }];
    d.create_bind_group(&BindGroupDesc {
        label: Some(n.into()),
        layout: l,
        entries: &e,
    })
}
pub(crate) async fn shader<B: OffscreenBackend>(
    d: &Device<B>,
    n: &str,
    b: &[u8],
    e: &str,
    s: ShaderStage,
) -> Result<Shader<B>> {
    d.shader(ShaderDesc {
        label: Some(n.into()),
        format: ShaderFormat::Dxil,
        bytes: Arc::from(b),
        entries: vec![ShaderEntryPoint {
            name: e.into(),
            stage: s,
            workgroup_size: None,
        }],
        required_features: vec![],
    })
    .await
}
pub(crate) async fn pipe<B: OffscreenBackend>(
    d: &Device<B>,
    n: &str,
    l: PipelineLayout<B>,
    f: TextureFormat,
    vs: &[u8],
    ps: &[u8],
) -> Result<RasterPipeline<B>> {
    let vs = shader(d, n, vs, "vs_main", ShaderStage::Vertex).await?;
    let ps = shader(d, n, ps, "ps_main", ShaderStage::Fragment).await?;
    let face = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    d.raster_pipeline(RasterPipelineDesc {
        label: Some(n.into()),
        layout: l,
        vertex: PipelineStage {
            module: vs,
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps,
            entry: "ps_main".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![VertexBufferLayout {
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
        }],
        color_targets: vec![Some(ColorTargetState {
            format: f,
            blend: None,
            write_mask: 15,
        })],
        primitive: PrimitiveState {
            topology: PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: FrontFace::Ccw,
            cull_mode: CullMode::None,
            unclipped_depth: false,
            polygon_mode: PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: true,
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
pub(crate) fn depth<B: OffscreenBackend>(
    d: &Device<B>,
    e: [u32; 2],
) -> Result<(Texture<B>, TextureView<B>)> {
    if e.contains(&0) {
        return Err(Error::InvalidRange);
    }
    let t = d.create_texture(&TextureDesc {
        label: Some("07 depth".into()),
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
pub(crate) fn matrix(t: f32, mirror: bool) -> [u8; 256] {
    let (s, c) = t.sin_cos();
    let y = if mirror { -1.0 } else { 1.0 };
    let x = [
        0.55 * c,
        0.0,
        0.55 * s,
        0.0,
        0.0,
        0.55 * y,
        0.0,
        0.0,
        -0.55 * s,
        0.0,
        0.55 * c,
        0.0,
        0.0,
        -0.15,
        0.3,
        1.0,
    ];
    let mut b = [0; 256];
    for (i, v) in x.into_iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes())
    }
    b
}
