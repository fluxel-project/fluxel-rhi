//! DXIL shaders, bind-group layouts, and raster pipeline descriptions for example 03.
use super::*;
use std::sync::Arc;

const ROCK_VS: &[u8] = include_bytes!("../assets/03_instancing/rock_vs.dxil");
const ROCK_PS: &[u8] = include_bytes!("../assets/03_instancing/rock_ps.dxil");
const PLANET_VS: &[u8] = include_bytes!("../assets/03_instancing/planet_vs.dxil");
const PLANET_PS: &[u8] = include_bytes!("../assets/03_instancing/planet_ps.dxil");
const STAR_VS: &[u8] = include_bytes!("../assets/03_instancing/star_vs.dxil");
const STAR_PS: &[u8] = include_bytes!("../assets/03_instancing/star_ps.dxil");

pub(crate) async fn scene_pipelines<B: InstancingBackend>(
    device: &Device<B>,
    format: TextureFormat,
) -> Result<(
    RasterPipeline<B>,
    RasterPipeline<B>,
    RasterPipeline<B>,
    BindGroupLayout<B>,
    BindGroupLayout<B>,
)> {
    let rock_layout = texture_layout(
        device,
        TextureViewDimension::D2Array,
        "03 rock texture layout",
    )?;
    let rock_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
        label: Some("03 rock scene layout".into()),
        groups: &[&rock_layout],
        push_constants: &[],
    })?;
    let rocks = pipeline(
        device,
        "rocks",
        rock_pipeline_layout,
        format,
        mesh(true),
        ROCK_VS,
        ROCK_PS,
        true,
    )
    .await?;
    let planet_layout =
        texture_layout(device, TextureViewDimension::D2, "03 planet texture layout")?;
    let planet_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
        label: Some("03 planet scene layout".into()),
        groups: &[&planet_layout],
        push_constants: &[],
    })?;
    let planet = pipeline(
        device,
        "planet",
        planet_pipeline_layout,
        format,
        mesh(false),
        PLANET_VS,
        PLANET_PS,
        true,
    )
    .await?;
    let stars_layout = device.create_pipeline_layout(&PipelineLayoutDesc {
        label: Some("03 stars layout".into()),
        groups: &[],
        push_constants: &[],
    })?;
    let stars = pipeline(
        device,
        "stars",
        stars_layout,
        format,
        vec![],
        STAR_VS,
        STAR_PS,
        false,
    )
    .await?;
    Ok((rocks, planet, stars, rock_layout, planet_layout))
}

pub(crate) fn texture_layout<B: InstancingBackend>(
    device: &Device<B>,
    dimension: TextureViewDimension,
    label: &str,
) -> Result<BindGroupLayout<B>> {
    device.create_bind_group_layout(&BindGroupLayoutDesc {
        label: Some(label.into()),
        entries: vec![
            binding(
                0,
                vec![ShaderStage::Vertex],
                BindingType::UniformBuffer {
                    dynamic_offset: false,
                    min_size: 160,
                },
            ),
            binding(
                1,
                vec![ShaderStage::Fragment],
                BindingType::SampledTexture {
                    sample: TextureSampleType::Float,
                    dimension,
                    multisampled: false,
                },
            ),
            binding(
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

async fn shader<B: InstancingBackend>(
    device: &Device<B>,
    label: &str,
    bytes: &[u8],
    entry: &str,
    stage: ShaderStage,
) -> Result<Shader<B>> {
    device
        .shader(ShaderDesc {
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
async fn pipeline<B: InstancingBackend>(
    device: &Device<B>,
    label: &str,
    layout: PipelineLayout<B>,
    format: TextureFormat,
    vertex_buffers: Vec<VertexBufferLayout>,
    vertex_dxil: &[u8],
    fragment_dxil: &[u8],
    depth: bool,
) -> Result<RasterPipeline<B>> {
    let (vertex_entry, fragment_entry) = match label {
        "rocks" => ("rock_vs", "rock_ps"),
        "planet" => ("planet_vs", "planet_ps"),
        _ => ("star_vs", "star_ps"),
    };
    let face = StencilFaceState {
        compare: CompareFunction::Always,
        fail: StencilOperation::Keep,
        depth_fail: StencilOperation::Keep,
        pass: StencilOperation::Keep,
    };
    device
        .raster_pipeline(RasterPipelineDesc {
            label: Some(label.into()),
            layout,
            vertex: PipelineStage {
                module: shader(
                    device,
                    label,
                    vertex_dxil,
                    vertex_entry,
                    ShaderStage::Vertex,
                )
                .await?,
                entry: vertex_entry.into(),
                constants: vec![],
            },
            fragment: Some(PipelineStage {
                module: shader(
                    device,
                    label,
                    fragment_dxil,
                    fragment_entry,
                    ShaderStage::Fragment,
                )
                .await?,
                entry: fragment_entry.into(),
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
                depth_write_enabled: label != "stars",
                depth_compare: CompareFunction::LessEqual,
                stencil_front: face,
                stencil_back: face,
                stencil_read_mask: 0,
                stencil_write_mask: 0,
                depth_bias_constant: 0,
                depth_bias_slope_scale: 0.,
                depth_bias_clamp: 0.,
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
fn mesh(instanced: bool) -> Vec<VertexBufferLayout> {
    let mut layouts = vec![VertexBufferLayout {
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
        layouts.push(VertexBufferLayout {
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
                    format: VertexFormat::Float32(1),
                },
            ],
        });
    }
    layouts
}
