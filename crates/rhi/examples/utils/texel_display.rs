//! Exact texel presentation of the ray tracing output.

use super::*;
use std::sync::Arc;

pub struct Display<B: PipelineBackend + RasterBackend> {
    pipeline: RasterPipeline<B>,
    group: BindGroup<B>,
}

impl<B: PipelineBackend + RasterBackend + ResourceBackend + ShaderBackend> Display<B> {
    pub async fn new(
        device: &Device<B>,
        view: &TextureView<B>,
        format: TextureFormat,
    ) -> Result<Self> {
        Self::new_with_format(device, view, format, ShaderFormat::Dxil).await
    }

    pub async fn new_with_format(
        device: &Device<B>,
        view: &TextureView<B>,
        format: TextureFormat,
        shader_format: ShaderFormat,
    ) -> Result<Self> {
        let group_layout = device.create_bind_group_layout(&BindGroupLayoutDesc {
            label: Some("ray output display".into()),
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::Float,
                    dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })?;
        let group = device.create_bind_group(&BindGroupDesc {
            label: Some("ray output display"),
            layout: &group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(view),
            }],
        })?;
        let layout = device.create_pipeline_layout(&PipelineLayoutDesc {
            label: Some("ray output display"),
            groups: &[&group_layout],
            push_constants: &[],
        })?;
        async fn shader<B: ShaderBackend>(
            device: &Device<B>,
            bytes: &[u8],
            entry: &str,
            stage: ShaderStage,
            shader_format: ShaderFormat,
        ) -> Result<Shader<B>> {
            device
                .shader(ShaderDesc {
                    label: Some("ray output display".into()),
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
            device,
            match shader_format {
                ShaderFormat::Dxil => {
                    include_bytes!("../assets/16_raytracing_hello/display_vs.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/16_raytracing_hello/display_vs.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("display shader format")),
            },
            "vs_main",
            ShaderStage::Vertex,
            shader_format,
        )
        .await?;
        let ps = shader(
            device,
            match shader_format {
                ShaderFormat::Dxil => {
                    include_bytes!("../assets/16_raytracing_hello/display_ps.dxil").as_slice()
                }
                ShaderFormat::SpirV => {
                    include_bytes!("../assets/16_raytracing_hello/display_ps.spv").as_slice()
                }
                _ => return Err(Error::Unsupported("display shader format")),
            },
            "ps_main",
            ShaderStage::Fragment,
            shader_format,
        )
        .await?;
        let stage = |module, entry: &str| PipelineStage {
            module,
            entry: entry.into(),
            constants: vec![],
        };
        let pipeline = device
            .raster_pipeline(RasterPipelineDesc {
                label: Some("ray output display".into()),
                layout,
                vertex: stage(vs, "vs_main"),
                fragment: Some(stage(ps, "ps_main")),
                vertex_buffers: vec![],
                color_targets: vec![Some(ColorTargetState {
                    format,
                    blend: None,
                    write_mask: 0xf,
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
        Ok(Self { pipeline, group })
    }

    pub fn encode(
        &self,
        encoder: &mut CommandEncoder<B>,
        image: &SurfaceImage<B>,
        extent: [u32; 2],
    ) -> Result<()> {
        let colors = [Some(ColorAttachment {
            view: AttachmentView::Surface(image),
            resolve_target: None,
            load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
            store: StoreOp::Store,
        })];
        let mut pass = encoder.begin_raster(&RasterPassDesc {
            label: Some("ray output display"),
            colors: &colors,
            depth_stencil: None,
            render_area: Rect {
                x: 0,
                y: 0,
                width: extent[0],
                height: extent[1],
            },
            multiview_mask: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            secondary_inheritance: None,
        })?;
        pass.viewport(Viewport {
            x: 0.0,
            y: 0.0,
            width: extent[0] as f32,
            height: extent[1] as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        })?;
        pass.scissor(Rect {
            x: 0,
            y: 0,
            width: extent[0],
            height: extent[1],
        })?;
        pass.set_pipeline(&self.pipeline)?;
        pass.set_bind_group(0, &self.group, &[])?;
        pass.draw(0..3, 0..1)?;
        pass.finish()
    }
}
