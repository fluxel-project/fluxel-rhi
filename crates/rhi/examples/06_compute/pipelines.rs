//! Shader module and pipeline construction for the compute-filter example.
use super::ComputeBackend;
use fluxel_rhi::api::*;
use std::sync::Arc;

const EMBOSS: &[u8] = include_bytes!("../assets/06_compute/emboss.dxil");
const EDGE: &[u8] = include_bytes!("../assets/06_compute/edgedetect.dxil");
const SHARPEN: &[u8] = include_bytes!("../assets/06_compute/sharpen.dxil");
const DISPLAY_VS: &[u8] = include_bytes!("../assets/06_compute/display_vs.dxil");
const DISPLAY_PS: &[u8] = include_bytes!("../assets/06_compute/display_ps.dxil");
const EMBOSS_SPIRV: &[u8] = include_bytes!("../assets/06_compute/emboss.spv");
const EDGE_SPIRV: &[u8] = include_bytes!("../assets/06_compute/edgedetect.spv");
const SHARPEN_SPIRV: &[u8] = include_bytes!("../assets/06_compute/sharpen.spv");
const DISPLAY_VS_SPIRV: &[u8] = include_bytes!("../assets/06_compute/display_vs.spv");
const DISPLAY_PS_SPIRV: &[u8] = include_bytes!("../assets/06_compute/display_ps.spv");
const FILTERS_WGSL: &[u8] = include_bytes!("../assets/06_compute/filters.wgsl");
const DISPLAY_WGSL: &[u8] = include_bytes!("../assets/06_compute/display.wgsl");

pub(super) async fn filters<B: ComputeBackend>(
    device: &Device<B>,
    layout: PipelineLayout<B>,
) -> Result<[ComputePipeline<B>; 3]> {
    Ok([
        compute_pipeline(device, layout.clone(), "emboss", EMBOSS, EMBOSS_SPIRV).await?,
        compute_pipeline(device, layout.clone(), "edgedetect", EDGE, EDGE_SPIRV).await?,
        compute_pipeline(device, layout, "sharpen", SHARPEN, SHARPEN_SPIRV).await?,
    ])
}

pub(super) async fn display<B: ComputeBackend>(
    device: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
) -> Result<RasterPipeline<B>> {
    let vs = shader(
        device,
        "06 display VS",
        "vs_main",
        ShaderStage::Vertex,
        DISPLAY_VS,
        DISPLAY_VS_SPIRV,
        DISPLAY_WGSL,
    )
    .await?;
    let ps = shader(
        device,
        "06 display PS",
        "ps_main",
        ShaderStage::Fragment,
        DISPLAY_PS,
        DISPLAY_PS_SPIRV,
        DISPLAY_WGSL,
    )
    .await?;
    device
        .raster_pipeline(RasterPipelineDesc {
            label: Some("06 display pipeline".into()),
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
            vertex_buffers: vec![VertexBufferLayout {
                stride: 20,
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
                        format: VertexFormat::Float32(2),
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
            depth_stencil: None,
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

async fn compute_pipeline<B: ComputeBackend>(
    device: &Device<B>,
    layout: PipelineLayout<B>,
    entry: &str,
    dxil: &[u8],
    spirv: &[u8],
) -> Result<ComputePipeline<B>> {
    let shader = shader(
        device,
        entry,
        entry,
        ShaderStage::Compute,
        dxil,
        spirv,
        FILTERS_WGSL,
    )
    .await?;
    device
        .compute()
        .ok_or(Error::Unsupported("compute"))?
        .create_pipeline(ComputePipelineDesc {
            label: Some(entry.into()),
            layout,
            compute: PipelineStage {
                module: shader,
                entry: entry.into(),
                constants: vec![],
            },
            required_subgroup_size: None,
            cache: None,
        })
        .await
}

async fn shader<B: ComputeBackend>(
    device: &Device<B>,
    label: &str,
    entry: &str,
    stage: ShaderStage,
    dxil: &[u8],
    spirv: &[u8],
    wgsl: &[u8],
) -> Result<Shader<B>> {
    for (format, bytes) in [
        (ShaderFormat::Dxil, dxil),
        (ShaderFormat::Wgsl, wgsl),
        (ShaderFormat::SpirV, spirv),
    ] {
        let desc = ShaderDesc {
            label: Some(label.into()),
            format,
            bytes: Arc::from(bytes),
            entries: vec![ShaderEntryPoint {
                name: entry.into(),
                stage,
                workgroup_size: (stage == ShaderStage::Compute).then_some([16, 16, 1]),
            }],
            required_features: vec![],
        };
        if !matches!(device.shader_route(&desc), ShaderRoute::Unsupported) {
            return device.shader(desc).await;
        }
    }
    Err(Error::Unsupported("06 shader route"))
}
