//! Create the reference model resources and mesh pipeline.
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
        let model = model::load(include_bytes!(
            "../assets/18_meshlet_instancing/ToyRobot.bin"
        ))
        .ok_or(Error::InvalidState)?;
        let sizes = [
            256,
            256,
            model.vertices.len(),
            model.meshlets.len(),
            model.unique_indices.len(),
            model.primitives.len(),
            128,
        ];
        let mut buffers = Vec::new();
        for (i, size) in sizes.into_iter().enumerate() {
            buffers.push(d.create_buffer(&BufferDesc {
                label: Some(format!("mesh model buffer {i}")),
                size: size as u64,
                usage: BufferUsage::COPY_DST
                    | if i < 2 {
                        BufferUsage::UNIFORM
                    } else {
                        BufferUsage::STORAGE
                    },
                memory: MemoryClass::Device,
            })?);
        }
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
        let mut entries = Vec::new();
        for (i, binding) in [0, 2, 3, 4, 5, 6, 7].into_iter().enumerate() {
            let ty = if i < 2 {
                BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: sizes[i] as u64,
                }
            } else {
                BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: sizes[i] as u64,
                    access: StorageAccess::Read,
                }
            };
            entries.push(BindGroupLayoutEntry {
                binding,
                visibility: if i == 0 {
                    vec![ShaderStage::Mesh, ShaderStage::Fragment]
                } else {
                    vec![ShaderStage::Mesh]
                },
                ty,
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            });
        }
        let group_layout = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("mesh resources".into()),
            entries,
        })?;
        let views = [
            None,
            None,
            Some(StorageBufferView::Structured { stride: 24 }),
            Some(StorageBufferView::Structured { stride: 16 }),
            Some(StorageBufferView::Raw),
            Some(StorageBufferView::Structured { stride: 4 }),
            Some(StorageBufferView::Structured { stride: 128 }),
        ];
        let entries: Vec<_> = [0, 2, 3, 4, 5, 6, 7]
            .into_iter()
            .enumerate()
            .map(|(i, binding)| BindGroupEntry {
                binding,
                resource: match views[i] {
                    Some(view) => BindingResource::StorageBufferView {
                        buffer: &buffers[i],
                        range: 0..sizes[i] as u64,
                        view,
                    },
                    None => BindingResource::Buffer {
                        buffer: &buffers[i],
                        range: 0..sizes[i] as u64,
                    },
                },
            })
            .collect();
        let group = d.create_bind_group(&BindGroupDesc {
            label: Some("mesh resources"),
            layout: &group_layout,
            entries: &entries,
        })?;
        let layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("mesh layout"),
            groups: &[&group_layout],
            push_constants: &[PushConstantRange {
                stages: vec![ShaderStage::Mesh],
                range: 0..8,
            }],
        })?;
        async fn shader<B: ShaderBackend>(
            d: &Device<B>,
            bytes: &[u8],
            stage: ShaderStage,
            shader_format: ShaderFormat,
        ) -> Result<Shader<B>> {
            d.shader(ShaderDesc {
                label: None,
                format: shader_format,
                bytes: Arc::from(bytes),
                entries: vec![ShaderEntryPoint {
                    name: "main".into(),
                    stage,
                    workgroup_size: None,
                }],
                required_features: vec![],
            })
            .await
        }
        let ms = shader(
            d,
            match shader_format {
                ShaderFormat::Dxil => {
                    include_bytes!("../assets/18_meshlet_instancing/MeshletMS.rhi.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/18_meshlet_instancing/MeshletMS.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("mesh shader format")),
            },
            ShaderStage::Mesh,
            shader_format,
        )
        .await?;
        let ps = shader(
            d,
            match shader_format {
                ShaderFormat::Dxil => {
                    include_bytes!("../assets/18_meshlet_instancing/MeshletPS.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/18_meshlet_instancing/MeshletPS.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("mesh shader format")),
            },
            ShaderStage::Fragment,
            shader_format,
        )
        .await?;
        let face = StencilFaceState {
            compare: CompareFunction::Always,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::Keep,
            pass: StencilOperation::Keep,
        };
        let pipeline = d
            .mesh()
            .ok_or(Error::Unsupported("mesh shaders"))?
            .create_pipeline(MeshPipelineDesc {
                layout,
                task: None,
                mesh: PipelineStage {
                    module: ms,
                    entry: "main".into(),
                    constants: vec![],
                },
                fragment: Some(PipelineStage {
                    module: ps,
                    entry: "main".into(),
                    constants: vec![],
                }),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Cw,
                    cull_mode: CullMode::Back,
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                targets: vec![Some(ColorTargetState {
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
                cache: None,
            })
            .await?;
        let display = display::Display::new_with_format(d, &view, format, shader_format).await?;
        Ok(Self {
            lane,
            buffers,
            output,
            view,
            depth,
            depth_view,
            readback,
            row_pitch,
            group,
            pipeline,
            display,
            initialized: false,
        })
    }
}
