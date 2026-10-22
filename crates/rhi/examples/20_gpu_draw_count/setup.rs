//! Create the GPU draw-count resources and pipelines.
use super::*;
use std::sync::Arc;

impl<B: ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + ReadbackBackend>
    Demo<B>
{
    pub(super) async fn new(
        d: &Device<B>,
        lane: LaneId,
        extent: [u32; 2],
        format: TextureFormat,
        shader_format: ShaderFormat,
    ) -> Result<Self> {
        if extent != [1280, 720] {
            return Err(Error::Unsupported("reference extent is 1280x720"));
        }
        let buffer = |label: &str, size, usage| {
            d.create_buffer(&BufferDesc {
                label: Some(label.into()),
                size,
                usage,
                memory: MemoryClass::Device,
            })
        };
        let scene = buffer(
            "triangle scene",
            262144,
            BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::VERTEX,
        )?;
        let vertex = buffer(
            "triangle vertex",
            36,
            BufferUsage::VERTEX | BufferUsage::COPY_DST,
        )?;
        let draws = buffer(
            "generated draws",
            16384,
            BufferUsage::STORAGE | BufferUsage::INDIRECT | BufferUsage::COPY_SRC,
        )?;
        let count = buffer(
            "generated count",
            4,
            BufferUsage::STORAGE | BufferUsage::INDIRECT | BufferUsage::COPY_SRC,
        )?;
        let count_readback = d.create_buffer(&BufferDesc {
            label: Some("count readback".into()),
            size: 4,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })?;
        let draws_readback = d.create_buffer(&BufferDesc {
            label: Some("draws readback".into()),
            size: 16384,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })?;
        let output = d.create_texture(&TextureDesc {
            label: Some("mesh output".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Rgba8Unorm,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT
                | TextureUsage::TEXTURE_BINDING
                | TextureUsage::COPY_SRC,
            view_formats: vec![],
        })?;
        let view = d.create_view(
            &output,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let depth = d.create_texture(&TextureDesc {
            label: Some("mesh depth".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })?;
        let depth_view = d.create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth32Float,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let row_pitch = extent[0] * 4;
        let readback = d.create_buffer(&BufferDesc {
            label: Some("mesh readback".into()),
            size: u64::from(row_pitch) * u64::from(extent[1]),
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })?;
        let binding = |binding, stage, access, min_size| BindGroupLayoutEntry {
            binding,
            visibility: vec![stage],
            ty: BindingType::StorageBuffer {
                dynamic_offset: false,
                min_size,
                access,
            },
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        };
        let compute_layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("culling resources".into()),
            entries: vec![
                binding(0, ShaderStage::Compute, StorageAccess::Read, 262144),
                binding(1, ShaderStage::Compute, StorageAccess::Write, 16384),
                binding(2, ShaderStage::Compute, StorageAccess::Write, 4),
            ],
        })?;
        let compute_group = d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &compute_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::StorageBufferView {
                        buffer: &scene,
                        range: 0..262144,
                        view: StorageBufferView::Structured { stride: 256 },
                    },
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::StorageBufferView {
                        buffer: &draws,
                        range: 0..16384,
                        view: StorageBufferView::Structured { stride: 16 },
                    },
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::StorageBufferView {
                        buffer: &count,
                        range: 0..4,
                        view: StorageBufferView::Raw,
                    },
                },
            ],
        })?;
        let compute_pipeline_layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&compute_layout],
            push_constants: &[],
        })?;
        let layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })?;
        async fn shader<B: ShaderBackend>(
            d: &Device<B>,
            bytes: &[u8],
            name: &str,
            stage: ShaderStage,
            shader_format: ShaderFormat,
        ) -> Result<Shader<B>> {
            d.shader(ShaderDesc {
                label: None,
                format: shader_format,
                bytes: Arc::from(bytes),
                entries: vec![ShaderEntryPoint {
                    name: name.into(),
                    stage,
                    workgroup_size: None,
                }],
                required_features: vec![],
            })
            .await
        }
        let cs = shader(
            d,
            match shader_format {
                ShaderFormat::Dxil => {
                    include_bytes!("../assets/20_gpu_draw_count/cull.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/20_gpu_draw_count/cull.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("example shader format")),
            },
            "CSMain",
            ShaderStage::Compute,
            shader_format,
        )
        .await?;
        let vs = shader(
            d,
            match shader_format {
                ShaderFormat::Dxil => {
                    include_bytes!("../assets/20_gpu_draw_count/draw_vs.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/20_gpu_draw_count/draw_vs.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("example shader format")),
            },
            "VSMain",
            ShaderStage::Vertex,
            shader_format,
        )
        .await?;
        let ps = shader(
            d,
            match shader_format {
                ShaderFormat::Dxil => {
                    include_bytes!("../assets/20_gpu_draw_count/draw_ps.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/20_gpu_draw_count/draw_ps.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("example shader format")),
            },
            "PSMain",
            ShaderStage::Fragment,
            shader_format,
        )
        .await?;
        let compute_pipeline = d
            .compute()
            .ok_or(Error::Unsupported("compute"))?
            .create_pipeline(ComputePipelineDesc {
                label: None,
                layout: compute_pipeline_layout,
                compute: PipelineStage {
                    module: cs,
                    entry: "CSMain".into(),
                    constants: vec![],
                },
                required_subgroup_size: None,
                cache: None,
            })
            .await?;
        let face = StencilFaceState {
            compare: CompareFunction::Always,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::Keep,
            pass: StencilOperation::Keep,
        };
        let pipeline = d
            .raster_pipeline(RasterPipelineDesc {
                label: Some("indirect triangles".into()),
                layout,
                vertex: PipelineStage {
                    module: vs,
                    entry: "VSMain".into(),
                    constants: vec![],
                },
                fragment: Some(PipelineStage {
                    module: ps,
                    entry: "PSMain".into(),
                    constants: vec![],
                }),
                vertex_buffers: vec![
                    VertexBufferLayout {
                        stride: 12,
                        step_mode: VertexStepMode::Vertex,
                        attributes: vec![VertexAttribute {
                            location: 0,
                            offset: 0,
                            format: VertexFormat::Float32(3),
                        }],
                    },
                    VertexBufferLayout {
                        stride: 256,
                        step_mode: VertexStepMode::Instance,
                        attributes: (0..6)
                            .map(|i| VertexAttribute {
                                location: i + 1,
                                offset: 16 + u64::from(i) * 16,
                                format: VertexFormat::Float32(4),
                            })
                            .collect(),
                    },
                ],
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Cw,
                    cull_mode: CullMode::Back,
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                color_targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: 15,
                })],
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::Less,
                    stencil_front: face,
                    stencil_back: face,
                    stencil_read_mask: 255,
                    stencil_write_mask: 255,
                    depth_bias_constant: 0,
                    depth_bias_slope_scale: 0.0,
                    depth_bias_clamp: 0.0,
                }),
                multisample: MultisampleState {
                    count: 1,
                    mask: u32::MAX as u64,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
            .await?;
        let display = display::Display::new_with_format(d, &view, format, shader_format).await?;
        Ok(Self {
            lane,
            scene,
            vertex,
            draws,
            count,
            count_readback,
            draws_readback,
            compute_group,
            compute_pipeline,
            output,
            view,
            depth,
            depth_view,
            readback,
            row_pitch,
            pipeline,
            display,
            initialized: false,
        })
    }
}
