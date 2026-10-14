//! Multiview attachment, transition, pipeline, and stereo-uniform helpers.

use super::*;

pub(super) fn layered_target<B: MultiviewBackend>(
    device: &Device<B>,
    format: TextureFormat,
    extent: [u32; 2],
) -> Result<(Texture<B>, TextureView<B>)> {
    let texture = device.create_texture(&TextureDesc {
        label: Some("15 two eye attachment".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 2,
        format,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT.union(TextureUsage::TEXTURE_BINDING),
        view_formats: vec![],
    })?;
    let view = device.create_view(
        &texture,
        &TextureViewDesc {
            format,
            dimension: TextureViewDimension::D2Array,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..2,
        },
    )?;
    Ok((texture, view))
}
pub(super) fn layered_depth<B: MultiviewBackend>(
    device: &Device<B>,
    extent: [u32; 2],
) -> Result<(Texture<B>, TextureView<B>)> {
    let texture = device.create_texture(&TextureDesc {
        label: Some("15 two eye depth".into()),
        dimension: TextureDimension::D2,
        extent: [extent[0], extent[1], 1],
        array_layers: 2,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    })?;
    let view = device.create_view(
        &texture,
        &TextureViewDesc {
            format: TextureFormat::Depth32Float,
            dimension: TextureViewDimension::D2Array,
            aspects: TextureAspects::DEPTH,
            mips: 0..1,
            layers: 0..2,
        },
    )?;
    Ok((texture, view))
}
pub(super) fn binding(
    binding: u32,
    ty: BindingType,
    visibility: ShaderStage,
) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![visibility],
        ty,
        count: BindingCount::One,
        partially_bound: false,
        nonuniform_indexing: false,
    }
}
pub(super) fn transition<'a, B: MultiviewBackend>(
    texture: &'a Texture<B>,
    from: ImageLayout,
    to: ImageLayout,
    before: TextureUses,
    after: TextureUses,
) -> ResourceTransition<'a, B> {
    ResourceTransition::Texture(TextureBarrier {
        target: Subresources {
            texture,
            aspects: TextureAspects::COLOR,
            mips: 0..1,
            layers: 0..2,
        },
        usage: StateTransition {
            from: before,
            to: after,
        },
        layout: Some((from, to)),
        ownership: None,
    })
}
pub(super) async fn pipeline<B: MultiviewBackend>(
    device: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
    vs: &[u8],
    ps: &[u8],
    vs_spirv: &[u8],
    ps_spirv: &[u8],
    multiview: bool,
    depth: bool,
    vertex_buffers: Vec<VertexBufferLayout>,
) -> Result<RasterPipeline<B>> {
    let vs = utils::native_shader(
        device,
        "15 VS",
        "vs_main",
        ShaderStage::Vertex,
        vs,
        vs_spirv,
    )
    .await?;
    let ps = utils::native_shader(
        device,
        "15 PS",
        "ps_main",
        ShaderStage::Fragment,
        ps,
        ps_spirv,
    )
    .await?;
    device
        .raster_pipeline(RasterPipelineDesc {
            label: Some("15 raster pipeline".into()),
            layout,
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
                cull_mode: CullMode::None,
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: depth.then_some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: CompareFunction::LessEqual,
                stencil_front: StencilFaceState {
                    compare: CompareFunction::Always,
                    fail: StencilOperation::Keep,
                    depth_fail: StencilOperation::Keep,
                    pass: StencilOperation::Keep,
                },
                stencil_back: StencilFaceState {
                    compare: CompareFunction::Always,
                    fail: StencilOperation::Keep,
                    depth_fail: StencilOperation::Keep,
                    pass: StencilOperation::Keep,
                },
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
            multiview: multiview.then_some(2),
            cache: None,
        })
        .await
}
pub(super) fn two_eye_matrices(extent: [u32; 2]) -> [u8; 272] {
    // Same stereo frusta and first-person camera as the C++ multiview sample:
    // 90° FOV, near/far 0.1/256, 0.08 eye separation, camera (7, 3.2, 0)
    // rotated +90° around Y. Matrices are column-major for HLSL `mul(M, v)`.
    let aspect = extent[0].max(1) as f32 * 0.5 / extent[1].max(1) as f32;
    let near = 0.1;
    let far = 256.0;
    let half_width = near; // tan(90° / 2) * near
    let eye_shift = 0.08 * (near / 0.5) * 0.5;
    let projection = |left: f32, right: f32| {
        [
            2.0 * near / (right - left),
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            (right + left) / (right - left),
            0.0,
            far / (near - far),
            -1.0,
            0.0,
            0.0,
            far * near / (near - far),
            0.0,
        ]
    };
    let modelview = |z: f32| {
        // `rotY(90°) * translate(7, 3.2, z)` from Camera::updateViewMatrix.
        [
            0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, z, 3.2, -7.0, 1.0,
        ]
    };
    let mut bytes = [0; 272];
    for (index, value) in projection(
        -aspect * half_width - eye_shift,
        aspect * half_width - eye_shift,
    )
    .into_iter()
    .chain(projection(
        -aspect * half_width + eye_shift,
        aspect * half_width + eye_shift,
    ))
    .chain(modelview(0.04))
    .chain(modelview(-0.04))
    .chain([-2.5, -3.5, 0.0, 1.0])
    .enumerate()
    {
        bytes[index * 4..index * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    }
    bytes
}
