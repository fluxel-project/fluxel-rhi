//! Regression checks for the owning private GL boundary.
use super::*;

#[test]
fn depth_bias_uses_gl_factor_units_order() {
    let mut depth = crate::backend::api::GlDepthStencilState {
        depth_write_enabled: true,
        depth_compare: crate::backend::api::GlDepthCompareFunction::Less,
        depth_bias: [0; 3],
        stencil_front: stencil_face(),
        stencil_back: stencil_face(),
        stencil_reference: 0,
    };
    apply_depth_bias(
        &mut depth,
        DynamicDepthBias {
            constant: 3,
            slope_scale: 2.0,
            clamp: 1.0,
        },
    );
    assert_eq!(
        depth.depth_bias,
        [2.0f32.to_bits(), 3.0f32.to_bits(), 1.0f32.to_bits()]
    );
}

fn stencil_face() -> crate::backend::api::GlStencilFaceState {
    crate::backend::api::GlStencilFaceState {
        compare: crate::backend::api::GlDepthCompareFunction::Always,
        fail_op: crate::backend::api::GlStencilOperation::Keep,
        depth_fail_op: crate::backend::api::GlStencilOperation::Keep,
        pass_op: crate::backend::api::GlStencilOperation::Keep,
        read_mask: 0,
        write_mask: 0,
    }
}
