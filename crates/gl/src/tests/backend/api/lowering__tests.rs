
use super::*;

#[test]
fn fixed_raster_mappings_cover_every_api_variant() {
    assert_eq!(
        [
            raster_topology_mode(GlPrimitiveTopology::Points),
            raster_topology_mode(GlPrimitiveTopology::Lines),
            raster_topology_mode(GlPrimitiveTopology::LineStrip),
            raster_topology_mode(GlPrimitiveTopology::Triangles),
            raster_topology_mode(GlPrimitiveTopology::TriangleStrip),
        ],
        [0x0000, 0x0001, 0x0003, 0x0004, 0x0005]
    );
    assert_eq!(
        [
            raster_compare_function(GlDepthCompareFunction::Never),
            raster_compare_function(GlDepthCompareFunction::Less),
            raster_compare_function(GlDepthCompareFunction::Equal),
            raster_compare_function(GlDepthCompareFunction::LessEqual),
            raster_compare_function(GlDepthCompareFunction::Greater),
            raster_compare_function(GlDepthCompareFunction::NotEqual),
            raster_compare_function(GlDepthCompareFunction::GreaterEqual),
            raster_compare_function(GlDepthCompareFunction::Always),
        ],
        [
            0x0200, 0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207
        ]
    );
    assert_eq!(
        [
            raster_stencil_operation(GlStencilOperation::Keep),
            raster_stencil_operation(GlStencilOperation::Zero),
            raster_stencil_operation(GlStencilOperation::Replace),
            raster_stencil_operation(GlStencilOperation::IncrementClamp),
            raster_stencil_operation(GlStencilOperation::DecrementClamp),
            raster_stencil_operation(GlStencilOperation::Invert),
            raster_stencil_operation(GlStencilOperation::IncrementWrap),
            raster_stencil_operation(GlStencilOperation::DecrementWrap),
        ],
        [0x1E00, 0, 0x1E01, 0x1E02, 0x1E03, 0x150A, 0x8507, 0x8508]
    );
    assert_eq!(
        [
            raster_blend_factor(GlBlendFactor::Zero),
            raster_blend_factor(GlBlendFactor::One),
            raster_blend_factor(GlBlendFactor::Src),
            raster_blend_factor(GlBlendFactor::OneMinusSrc),
            raster_blend_factor(GlBlendFactor::SrcAlpha),
            raster_blend_factor(GlBlendFactor::OneMinusSrcAlpha),
            raster_blend_factor(GlBlendFactor::Dst),
            raster_blend_factor(GlBlendFactor::OneMinusDst),
            raster_blend_factor(GlBlendFactor::DstAlpha),
            raster_blend_factor(GlBlendFactor::OneMinusDstAlpha),
            raster_blend_factor(GlBlendFactor::Constant),
            raster_blend_factor(GlBlendFactor::OneMinusConstant),
            raster_blend_factor(GlBlendFactor::SrcAlphaSaturated),
            raster_blend_factor(GlBlendFactor::Src1),
            raster_blend_factor(GlBlendFactor::OneMinusSrc1),
            raster_blend_factor(GlBlendFactor::Src1Alpha),
            raster_blend_factor(GlBlendFactor::OneMinusSrc1Alpha),
        ],
        [
            0, 1, 0x0300, 0x0301, 0x0302, 0x0303, 0x0306, 0x0307, 0x0304, 0x0305, 0x8001, 0x8002,
            0x0308, 0x88F9, 0x88FA, 0x8589, 0x88FB,
        ]
    );
    assert_eq!(
        [
            raster_blend_operation(GlBlendOperation::Add),
            raster_blend_operation(GlBlendOperation::Subtract),
            raster_blend_operation(GlBlendOperation::ReverseSubtract),
            raster_blend_operation(GlBlendOperation::Min),
            raster_blend_operation(GlBlendOperation::Max),
        ],
        [0x8006, 0x800A, 0x800B, 0x8007, 0x8008]
    );
    assert_eq!(
        [
            sampler_address_mode(GlAddressMode::ClampToEdge),
            sampler_address_mode(GlAddressMode::Repeat),
            sampler_address_mode(GlAddressMode::MirroredRepeat),
        ],
        [0x812F, 0x2901, 0x8370]
    );
    assert_eq!(
        [
            sampler_mag_filter(GlFilterMode::Nearest),
            sampler_mag_filter(GlFilterMode::Linear),
        ],
        [0x2600, 0x2601]
    );
    assert_eq!(
        [
            sampler_min_filter(GlFilterMode::Nearest, GlMipmapFilterMode::Nearest),
            sampler_min_filter(GlFilterMode::Linear, GlMipmapFilterMode::Nearest),
            sampler_min_filter(GlFilterMode::Nearest, GlMipmapFilterMode::Linear),
            sampler_min_filter(GlFilterMode::Linear, GlMipmapFilterMode::Linear),
        ],
        [0x2700, 0x2701, 0x2702, 0x2703]
    );
    assert_eq!(
        [
            sampler_compare_function(GlCompareFunction::Never),
            sampler_compare_function(GlCompareFunction::Less),
            sampler_compare_function(GlCompareFunction::Equal),
            sampler_compare_function(GlCompareFunction::LessEqual),
            sampler_compare_function(GlCompareFunction::Greater),
            sampler_compare_function(GlCompareFunction::NotEqual),
            sampler_compare_function(GlCompareFunction::GreaterEqual),
            sampler_compare_function(GlCompareFunction::Always),
        ],
        [
            0x0200, 0x0201, 0x0202, 0x0203, 0x0204, 0x0205, 0x0206, 0x0207
        ]
    );
    assert_eq!(
        [
            depth_attachment_point(GlFormat::Depth16Unorm),
            depth_attachment_point(GlFormat::Depth24Unorm),
            depth_attachment_point(GlFormat::Depth32Float),
            depth_attachment_point(GlFormat::Depth24PlusStencil8),
            depth_attachment_point(GlFormat::Depth32FloatStencil8),
            depth_attachment_point(GlFormat::Stencil8),
            depth_attachment_point(GlFormat::Rgba8Unorm),
        ],
        [
            Some(0x8D00),
            Some(0x8D00),
            Some(0x8D00),
            Some(0x821A),
            Some(0x821A),
            Some(0x8D20),
            None,
        ]
    );
    assert_eq!(
        [
            has_stencil_plane(GlFormat::Depth24PlusStencil8),
            has_stencil_plane(GlFormat::Depth32FloatStencil8),
            has_stencil_plane(GlFormat::Stencil8),
            has_stencil_plane(GlFormat::Depth24Unorm),
            has_stencil_plane(GlFormat::Rgba8Unorm),
        ],
        [true, true, true, false, false]
    );
}
