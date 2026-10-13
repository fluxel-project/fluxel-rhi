//! Pipeline and camera data construction for the push-constant example.
//!
//! This module has no frame ownership. It builds descriptors and bytes consumed
//! by the workload recorder.
use super::*;

pub(super) fn buffer<B: PushBackend>(
    d: &Device<B>,
    label: &'static str,
    size: u64,
    usage: BufferUsage,
) -> Result<Buffer<B>> {
    d.create_buffer(&BufferDesc {
        label: Some(label.into()),
        size,
        usage,
        memory: MemoryClass::Device,
    })
}
async fn shader<B: PushBackend>(
    d: &Device<B>,
    dxil: &[u8],
    spirv: &[u8],
    entry: &str,
    stage: ShaderStage,
) -> Result<Shader<B>> {
    common::native_shader_with_wgsl(d, "05 shader", entry, stage, dxil, spirv, WGSL).await
}
pub(super) async fn create_pipeline<B: PushBackend>(
    d: &Device<B>,
    layout: PipelineLayout<B>,
    format: TextureFormat,
    dynamic_uniforms: bool,
) -> Result<RasterPipeline<B>> {
    let vs = shader(d, VERTEX_DXIL, VERTEX_SPIRV, "vs_main", ShaderStage::Vertex).await?;
    let ps = shader(
        d,
        FRAGMENT_DXIL,
        FRAGMENT_SPIRV,
        "fs_main",
        ShaderStage::Fragment,
    )
    .await?;
    d.raster_pipeline(RasterPipelineDesc {
        label: Some(
            if dynamic_uniforms {
                "05 WebGPU dynamic-uniform pipeline"
            } else {
                "05 push pipeline"
            }
            .into(),
        ),
        layout,
        vertex: PipelineStage {
            module: vs,
            entry: "vs_main".into(),
            constants: vec![],
        },
        fragment: Some(PipelineStage {
            module: ps,
            entry: "fs_main".into(),
            constants: vec![],
        }),
        vertex_buffers: vec![VertexBufferLayout {
            stride: gltf::VKGLTF_VERTEX_STRIDE as u64,
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
                    offset: 32,
                    format: VertexFormat::Float32(4),
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
        multiview: None,
        cache: None,
    })
    .await
}
pub(super) fn camera_bytes(extent: [u32; 2]) -> [u8; 256] {
    // The source uses a 60° camera at z=-10 and a uniform 0.5 model scale.
    let aspect = extent[0].max(1) as f32 / extent[1].max(1) as f32;
    let f = 1.0 / 30f32.to_radians().tan();
    let projection = [
        f / aspect,
        0.,
        0.,
        0.,
        0.,
        f,
        0.,
        0.,
        0.,
        0.,
        256. / 255.9,
        1.,
        0.,
        0.,
        -25.6 / 255.9,
        0.,
    ];
    let view = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 10., 1.,
    ];
    let scale: [f32; 16] = [
        0.5, 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 0.5, 0., 0., 0., 0., 1.,
    ];
    let projection_view = multiply(projection, view);
    let mut out = [0; 256];
    for (index, value) in projection_view.into_iter().enumerate() {
        out[index * 4..index * 4 + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (index, value) in scale.into_iter().enumerate() {
        out[64 + index * 4..68 + index * 4].copy_from_slice(&value.to_le_bytes());
    }
    out
}

fn multiply(a: [f32; 16], b: [f32; 16]) -> [f32; 16] {
    let mut output = [0.; 16];
    for column in 0..4 {
        for row in 0..4 {
            output[column * 4 + row] = (0..4).map(|k| a[k * 4 + row] * b[column * 4 + k]).sum();
        }
    }
    output
}
