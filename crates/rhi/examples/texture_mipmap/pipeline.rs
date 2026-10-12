use super::{MipChain, MipmapBackend};
use fluxel_rhi::api::*;

pub struct Tunnel<B: MipmapBackend> {
    pub(super) lane: LaneId,
    pub(super) chain: MipChain<B>,
    pub(super) vertices: Buffer<B>,
    pub(super) indices: Buffer<B>,
    pub(super) uniform: Buffer<B>,
    pub(super) depth: Texture<B>,
    pub(super) depth_view: TextureView<B>,
    pub(super) depth_extent: [u32; 2],
    pub(super) group: BindGroup<B>,
    pub(super) pipeline: RasterPipeline<B>,
    pub(super) vertex_data: Vec<u8>,
    pub(super) index_data: Vec<u8>,
    pub(super) index_count: u32,
    pub(super) pixels: Vec<u8>,
    pub(super) initialized: bool,
}
impl<B: MipmapBackend> Tunnel<B> {
    pub async fn new(
        device: &Device<B>,
        lane: LaneId,
        format: TextureFormat,
        extent: [u32; 2],
    ) -> Result<Self> {
        let image = super::resources::parse_ktx(crate::IMAGE)?;
        let model = crate::gltf::load_embedded_model(
            crate::MODEL,
            crate::gltf::LoadOptions::PRETRANSFORM_FLIP_Y,
        )
        .map_err(|error| Error::Backend(error.to_string()))?;
        let chain = MipChain::new(device, lane, image.width, image.height)?;
        let view = device.create_view(
            chain.texture(),
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..chain.levels(),
                layers: 0..1,
            },
        )?;
        let sampler = device.create_sampler(&SamplerDesc {
            label: Some("09 mip sampler".into()),
            address: [AddressMode::Repeat; 3],
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mip_filter: FilterMode::Linear,
            lod: 0.0..chain.levels() as f32,
            anisotropy: 1,
            compare: None,
            border: None,
        })?;
        let vertices = device.create_buffer(&buffer(
            "09 tunnel vertices",
            model.vertices.len() as u64,
            BufferUsage::VERTEX.union(BufferUsage::COPY_DST),
        ))?;
        let indices = device.create_buffer(&buffer(
            "09 tunnel indices",
            model.indices.len() as u64,
            BufferUsage::INDEX.union(BufferUsage::COPY_DST),
        ))?;
        let uniform = device.create_buffer(&buffer(
            "09 transform",
            256,
            BufferUsage::UNIFORM.union(BufferUsage::COPY_DST),
        ))?;
        let depth = device.create_texture(&TextureDesc {
            label: Some("09 tunnel depth".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })?;
        let depth_view = device.create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth32Float,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        let entry = |binding, visibility, ty| BindGroupLayoutEntry {
            binding,
            visibility,
            ty,
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        };
        let layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("09 textured tunnel bindings".into()),
            entries: vec![
                entry(
                    0,
                    vec![ShaderStage::Vertex],
                    BindingType::UniformBuffer {
                        dynamic_offset: false,
                        min_size: 64,
                    },
                ),
                entry(
                    1,
                    vec![ShaderStage::Fragment],
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
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
        })?;
        let entries = [
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer {
                    buffer: &uniform,
                    range: 0..256,
                },
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::TextureView(&view),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Sampler(&sampler),
            },
        ];
        let group = device.create_bind_group(&BindGroupDesc {
            label: Some("09 textured tunnel group"),
            layout: &layout,
            entries: &entries,
        })?;
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("09 tunnel layout"),
            groups: &[&layout],
            push_constants: &[],
        })?;
        let vs = shader(
            device,
            "09 tunnel crate::VS",
            crate::VS,
            crate::VS_SPIRV,
            "vs_main",
            ShaderStage::Vertex,
        )
        .await?;
        let ps = shader(
            device,
            "09 tunnel crate::PS",
            crate::PS,
            crate::PS_SPIRV,
            "ps_main",
            ShaderStage::Fragment,
        )
        .await?;
        let face = StencilFaceState {
            compare: CompareFunction::Always,
            fail: StencilOperation::Keep,
            depth_fail: StencilOperation::Keep,
            pass: StencilOperation::Keep,
        };
        let pipeline = device
            .raster_pipeline(RasterPipelineDesc {
                label: Some("09 textured tunnel".into()),
                layout: pipeline_layout,
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
                            offset: 24,
                            format: VertexFormat::Float32(2),
                        },
                        VertexAttribute {
                            location: 2,
                            offset: 12,
                            format: VertexFormat::Float32(3),
                        },
                    ],
                }],
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
            .await?;
        let index_count = (model.indices.len() / 4) as u32;
        Ok(Self {
            lane,
            chain,
            vertices,
            indices,
            uniform,
            depth,
            depth_view,
            depth_extent: extent,
            group,
            pipeline,
            vertex_data: model.vertices,
            index_data: model.indices,
            index_count,
            pixels: image.pixels,
            initialized: false,
        })
    }

    pub(super) fn resize_depth(&mut self, device: &Device<B>, extent: [u32; 2]) -> Result<()> {
        if self.depth_extent == extent {
            return Ok(());
        }
        let depth = device.create_texture(&TextureDesc {
            label: Some("09 tunnel depth".into()),
            dimension: TextureDimension::D2,
            extent: [extent[0], extent[1], 1],
            array_layers: 1,
            format: TextureFormat::Depth32Float,
            mip_levels: 1,
            samples: 1,
            usage: TextureUsage::RENDER_ATTACHMENT,
            view_formats: vec![],
        })?;
        let depth_view = device.create_view(
            &depth,
            &TextureViewDesc {
                format: TextureFormat::Depth32Float,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::DEPTH,
                mips: 0..1,
                layers: 0..1,
            },
        )?;
        self.depth = depth;
        self.depth_view = depth_view;
        self.depth_extent = extent;
        Ok(())
    }
}

fn buffer(label: &str, size: u64, usage: BufferUsage) -> BufferDesc {
    BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    }
}
async fn shader<B: MipmapBackend>(
    device: &Device<B>,
    label: &str,
    dxil: &[u8],
    spirv: &[u8],
    entry: &str,
    stage: ShaderStage,
) -> Result<Shader<B>> {
    crate::common::native_shader(device, label, entry, stage, dxil, spirv).await
}
