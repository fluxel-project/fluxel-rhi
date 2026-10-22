//! Device resources for the official triangle and standalone clear.

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
        let vertex = d.create_buffer(&BufferDesc {
            label: Some("triangle vertices".into()),
            size: 84,
            usage: BufferUsage::VERTEX | BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })?;
        let output = d.create_texture(&TextureDesc {
            label: Some("triangle output".into()),
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
        let row_pitch = extent[0]
            .checked_mul(4)
            .and_then(|n| n.checked_add(255))
            .ok_or(Error::InvalidRange)?
            & !255;
        let readback = d.create_buffer(&BufferDesc {
            label: Some("triangle readback".into()),
            size: u64::from(row_pitch) * u64::from(extent[1]),
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })?;
        async fn shader<B: ShaderBackend>(
            d: &Device<B>,
            bytes: &[u8],
            entry: &str,
            stage: ShaderStage,
            shader_format: ShaderFormat,
        ) -> Result<Shader<B>> {
            d.shader(ShaderDesc {
                label: Some("Microsoft HelloTriangle shader".into()),
                format: shader_format,
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
        let vs = shader(
            d,
            match shader_format {
                ShaderFormat::Dxil => {
                    include_bytes!("../assets/22_clear_texture/triangle_vs.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/22_clear_texture/triangle_vs.spv").as_slice()
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
                    include_bytes!("../assets/22_clear_texture/triangle_ps.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/22_clear_texture/triangle_ps.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("example shader format")),
            },
            "PSMain",
            ShaderStage::Fragment,
            shader_format,
        )
        .await?;
        let layout = d.create_pipeline_layout(&PipelineLayoutDesc {
            label: None,
            groups: &[],
            push_constants: &[],
        })?;
        let pipeline = d
            .raster_pipeline(RasterPipelineDesc {
                label: Some("Microsoft HelloTriangle".into()),
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
                vertex_buffers: vec![VertexBufferLayout {
                    stride: 28,
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
                            format: VertexFormat::Float32(4),
                        },
                    ],
                }],
                color_targets: vec![Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
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
                depth_stencil: None,
                multisample: MultisampleState {
                    count: 1,
                    mask: u64::from(u32::MAX),
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            })
            .await?;
        let display = display::Display::new_with_format(d, &view, format, shader_format).await?;
        Ok(Self {
            lane,
            vertex,
            output,
            view,
            readback,
            row_pitch,
            pipeline,
            display,
            initialized: false,
        })
    }
}
