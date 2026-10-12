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
pub(crate) fn scene_matrix(time: f32, reflected: bool, extent: [u32; 2]) -> [u8; 256] {
    // Keep the C++ sample's look-at camera and model order.  The simplified
    // predecessor scaled the complete scene in clip space, which moved both
    // dragon and mirror away from their reference framing.
    let angle = time * 10f32.to_radians();
    let (s, c) = angle.sin_cos();
    let rotation_y = [c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 0., 0., 0., 1.];
    let translate = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., -1., 0., 1.,
    ];
    let model = if reflected {
        multiply(
            multiply(
                rotation_y,
                [
                    1., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
                ],
            ),
            translate,
        )
    } else {
        multiply(rotation_y, translate)
    };
    encode_matrix(multiply(camera(extent), model))
}

pub(crate) fn plane_matrix(extent: [u32; 2]) -> [u8; 256] {
    encode_matrix(camera(extent))
}

fn camera(extent: [u32; 2]) -> [f32; 16] {
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1.0 / (60f32.to_radians() * 0.5).tan();
    let near = 0.1;
    let far = 256.0;
    let projection = [
        f / aspect,
        0.,
        0.,
        0.,
        0.,
        -f,
        0.,
        0.,
        0.,
        0.,
        far / (near - far),
        -1.,
        0.,
        0.,
        far * near / (near - far),
        0.,
    ];
    let (s, c) = (-2.5f32.to_radians()).sin_cos();
    let rotate_x = [1., 0., 0., 0., 0., c, s, 0., 0., -s, c, 0., 0., 0., 0., 1.];
    let translate = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., -6., 1.,
    ];
    multiply(projection, multiply(translate, rotate_x))
}

fn multiply(left: [f32; 16], right: [f32; 16]) -> [f32; 16] {
    let mut out = [0.; 16];
    for column in 0..4 {
        for row in 0..4 {
            out[column * 4 + row] = (0..4)
                .map(|k| left[k * 4 + row] * right[column * 4 + k])
                .sum();
        }
    }
    out
}

fn encode_matrix(x: [f32; 16]) -> [u8; 256] {
    let mut b = [0; 256];
    for (i, v) in x.into_iter().enumerate() {
        b[i * 4..i * 4 + 4].copy_from_slice(&v.to_ne_bytes())
    }
    b
}
