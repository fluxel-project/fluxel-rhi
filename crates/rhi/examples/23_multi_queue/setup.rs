//! Create particle storage and compute/graphics pipelines.
use super::*;
use std::sync::Arc;
const SIZE: u64 = 320000;
impl<
    B: ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + ReadbackBackend
        + DeferredBackend,
> Demo<B>
{
    pub(super) async fn new(
        d: &Device<B>,
        extent: [u32; 2],
        format: TextureFormat,
        shader_format: ShaderFormat,
    ) -> Result<Self> {
        if extent != [1280, 720] {
            return Err(Error::Unsupported("reference extent is 1280x720"));
        }
        let lane = |class| {
            d.lanes()
                .iter()
                .find(|l| l.class == class)
                .map(|l| l.id)
                .ok_or(Error::Unsupported("required queue lane"))
        };
        let compute = lane(LaneClass::Compute)?;
        let graphics = lane(LaneClass::Graphics)?;
        let make_buffer = || {
            d.create_buffer(&BufferDesc {
                label: None,
                size: SIZE,
                usage: BufferUsage::STORAGE | BufferUsage::COPY_DST | BufferUsage::COPY_SRC,
                memory: MemoryClass::Device,
            })
        };
        let particles = [make_buffer()?, make_buffer()?];
        let entry = |binding, stage, access| BindGroupLayoutEntry {
            binding,
            visibility: vec![stage],
            ty: BindingType::StorageBuffer {
                dynamic_offset: false,
                min_size: SIZE,
                access,
            },
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        };
        let cl = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(0, ShaderStage::Compute, StorageAccess::Read),
                entry(1, ShaderStage::Compute, StorageAccess::Write),
            ],
        })?;
        let gl = d.create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![entry(0, ShaderStage::Vertex, StorageAccess::Read)],
        })?;
        let resource = |buffer| BindingResource::StorageBufferView {
            buffer,
            range: 0..SIZE,
            view: StorageBufferView::Structured { stride: 32 },
        };
        let cg = |i: usize| {
            d.create_bind_group(&BindGroupDesc {
                label: None,
                layout: &cl,
                entries: &[
                    BindGroupEntry {
                        binding: 0,
                        resource: resource(&particles[i]),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: resource(&particles[1 - i]),
                    },
                ],
            })
        };
        let compute_groups = [cg(0)?, cg(1)?];
        let dg = |i: usize| {
            d.create_bind_group(&BindGroupDesc {
                label: None,
                layout: &gl,
                entries: &[BindGroupEntry {
                    binding: 0,
                    resource: resource(&particles[i]),
                }],
            })
        };
        let draw_groups = [dg(0)?, dg(1)?];
        let cpl = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&cl],
            push_constants: &[],
        })?;
        let gpl = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[&gl],
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
                    include_bytes!("../assets/23_multi_queue/compute.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/23_multi_queue/compute.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("shader format")),
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
                    include_bytes!("../assets/23_multi_queue/draw_vs.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/23_multi_queue/draw_vs.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("shader format")),
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
                    include_bytes!("../assets/23_multi_queue/draw_ps.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/23_multi_queue/draw_ps.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("shader format")),
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
                layout: cpl,
                compute: PipelineStage {
                    module: cs,
                    entry: "CSMain".into(),
                    constants: vec![],
                },
                cache: None,
                required_subgroup_size: None,
            })
            .await?;
        let color = BlendComponent {
            src_factor: BlendFactor::SrcAlpha,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        };
        let alpha = BlendComponent {
            src_factor: BlendFactor::Zero,
            dst_factor: BlendFactor::Zero,
            operation: BlendOperation::Add,
        };
        let pipeline = d
            .raster_pipeline(RasterPipelineDesc {
                label: None,
                layout: gpl,
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
                vertex_buffers: vec![],
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Cw,
                    cull_mode: CullMode::None,
                    unclipped_depth: false,
                    polygon_mode: PolygonMode::Fill,
                    conservative: false,
                },
                color_targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: Some(BlendState { color, alpha }),
                    write_mask: 15,
                })],
                depth_stencil: None,
                multisample: MultisampleState {
                    count: 1,
                    mask: u32::MAX as u64,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
            .await?;
        let output = d.create_texture(&TextureDesc {
            label: None,
            dimension: TextureDimension::D2,
            extent: [1280, 720, 1],
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
        let readback = d.create_buffer(&BufferDesc {
            label: None,
            size: 1280 * 720 * 4,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })?;
        let particle_readback = d.create_buffer(&BufferDesc {
            label: None,
            size: SIZE,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })?;
        let display = display::Display::new_with_format(d, &view, format, shader_format).await?;
        Ok(Self {
            compute,
            graphics,
            particles,
            compute_groups,
            draw_groups,
            compute_pipeline,
            pipeline,
            output,
            view,
            readback,
            particle_readback,
            display,
            frame: 0,
        })
    }
}
