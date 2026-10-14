//! Shader and raster-pipeline construction for the multisample Voyager scene.
//!
//! Attachment allocation and per-frame recording are deliberately kept in
//! sibling modules so pipeline state has one owner.

use super::MsaaBackend;
use fluxel_rhi::api::*;

pub(super) async fn shader<B: MsaaBackend>(
    device: &Device<B>,
    label: &str,
    bytes: &[u8],
    entry: &str,
    stage: ShaderStage,
) -> Result<Shader<B>> {
    let spirv = match stage {
        ShaderStage::Vertex => super::super::VS_SPIRV,
        ShaderStage::Fragment => super::super::PS_SPIRV,
        _ => return Err(Error::InvalidState),
    };
    super::super::common::native_shader_with_wgsl(
        device,
        label,
        entry,
        stage,
        bytes,
        spirv,
        include_bytes!("../assets/08_msaa/msaa.wgsl"),
    )
    .await
}
pub(super) async fn pipeline<B: MsaaBackend>(
    device: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
    samples: u32,
) -> Result<RasterPipeline<B>> {
    let vs = shader(
        device,
        "08 voyager vertex shader",
        super::super::VS,
        "vs_main",
        ShaderStage::Vertex,
    )
    .await?;
    let ps = shader(
        device,
        "08 voyager fragment shader",
        super::super::PS,
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
    device
        .raster_pipeline(RasterPipelineDesc {
            label: Some("08 MSAA voyager pipeline".into()),
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
                format,
                blend: None,
                write_mask: 15,
            })],
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: CullMode::Back,
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
                count: samples,
                mask: u64::from(u32::MAX),
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
            cache: None,
        })
        .await
}
