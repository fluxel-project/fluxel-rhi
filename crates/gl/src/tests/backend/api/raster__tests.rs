
use super::*;
#[test]
fn indexed_draw_keeps_base_and_instance_offsets() {
    let draw = GlDrawCommand::Indexed(GlIndexedDraw {
        first_index: 3,
        index_count: 6,
        instance_count: 2,
    });
    assert!(matches!(
        draw,
        GlDrawCommand::Indexed(GlIndexedDraw {
            instance_count: 2,
            ..
        })
    ));
}
#[test]
fn advanced_offset_is_capability_gated() {
    let draw = GlAdvancedDrawCommand {
        draw: GlDrawCommand::NonIndexed(GlNonIndexedDraw {
            first_vertex: 0,
            vertex_count: 3,
            instance_count: 1,
        }),
        base_vertex: 0,
        first_instance: 2,
    };
    assert_eq!(
        draw.validate(GlAdvancedRasterCapabilities {
            base_vertex: true,
            first_instance: false
        }),
        Err(GlRasterValidationError::MissingAdvancedCapability)
    );
}
#[test]
fn excess_color_targets_are_not_write_mask_failures() {
    fn state_with_targets(count: usize) -> GlRasterState {
        GlRasterState {
            rasterization: Default::default(),
            topology: GlPrimitiveTopology::Triangles,
            cull_mode: GlCullMode::None,
            front_face: GlFrontFace::CounterClockwise,
            depth_stencil: None,
            color_targets: vec![
                GlColorTargetState {
                    write_mask: 0xf,
                    blend: None,
                };
                count
            ],
            multisample: GlMultisampleState {
                sample_count: 1,
                alpha_to_coverage_enabled: false,
                sample_mask: u64::MAX,
            },
            viewport: GlViewport {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
                min_depth: 0.0f32.to_bits(),
                max_depth: 1.0f32.to_bits(),
            },
            scissor: None,
            blend_constant: [0; 4],
        }
    }
    let info = GlRasterValidationInfo {
        width: 1,
        height: 1,
        max_samples: 4,
        max_color_targets: 1,
    };
    assert_eq!(
        state_with_targets(2).validate(info),
        Err(GlRasterValidationError::TooManyColorTargets)
    );
    assert_eq!(state_with_targets(1).validate(info), Ok(()));
}
