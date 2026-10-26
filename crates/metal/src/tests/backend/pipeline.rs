//! Unit tests for the corresponding implementation module.

use super::{
    full_sample_mask, supported_color_format, supported_depth_state, supported_sample_count,
};
use fluxel_rhi_core::api::{self, TextureFormat};

#[test]
fn raster_target_rejects_depth_and_nonimplemented_formats() {
    assert!(supported_color_format(TextureFormat::Rgba8Unorm));
    assert!(supported_color_format(TextureFormat::Bgra8UnormSrgb));
    assert!(!supported_color_format(TextureFormat::Depth32Float));
    assert!(supported_color_format(TextureFormat::Rgba32Float));
}

#[test]
fn depth_pipeline_accepts_depth32_and_combined_stencil_semantics() {
    let face = api::StencilFaceState {
        compare: api::CompareFunction::Always,
        fail: api::StencilOperation::Keep,
        depth_fail: api::StencilOperation::Keep,
        pass: api::StencilOperation::Keep,
    };
    let state = api::DepthStencilState {
        format: TextureFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: api::CompareFunction::LessEqual,
        stencil_front: face,
        stencil_back: face,
        stencil_read_mask: 0,
        stencil_write_mask: 0,
        depth_bias_constant: 0,
        depth_bias_slope_scale: 0.0,
        depth_bias_clamp: 0.0,
    };
    assert!(supported_depth_state(state));
    assert!(!supported_depth_state(api::DepthStencilState {
        stencil_write_mask: 1,
        ..state
    }));
    assert!(!supported_depth_state(api::DepthStencilState {
        format: TextureFormat::Rgba8Unorm,
        ..state
    }));
    assert!(supported_depth_state(api::DepthStencilState {
        format: TextureFormat::Depth32FloatStencil8,
        stencil_front: api::StencilFaceState {
            compare: api::CompareFunction::LessEqual,
            fail: api::StencilOperation::Zero,
            depth_fail: api::StencilOperation::IncrementWrap,
            pass: api::StencilOperation::Replace,
        },
        stencil_read_mask: 0x0f,
        stencil_write_mask: 0xf0,
        ..state
    }));
}

#[test]
fn multisample_subset_accepts_only_native_counts_and_full_masks() {
    assert!(supported_sample_count(1));
    assert!(supported_sample_count(4));
    assert!(!supported_sample_count(3));
    assert!(full_sample_mask(u64::from(u32::MAX), 4));
    assert!(full_sample_mask(u64::MAX, 4));
    assert!(!full_sample_mask(0x0f, 8));
}
