    use super::*;

    #[test]
    fn extension_selection_and_topology_constraints() {
        let mut state = GlRasterizationState {
            polygon_mode: GlPolygonMode::Fill,
            depth_clamp: false,
            conservative: true,
        };
        assert_eq!(
            ConservativeFlavor::select(true, true),
            ConservativeFlavor::Nv
        );
        assert_eq!(
            ConservativeFlavor::select(false, true).token(),
            Some(0x83fe)
        );
        assert!(!ConservativeFlavor::None.supports(state, GlPrimitiveTopology::Triangles));
        assert!(ConservativeFlavor::Intel.supports(state, GlPrimitiveTopology::Triangles));
        assert!(ConservativeFlavor::Intel.supports(state, GlPrimitiveTopology::TriangleStrip));
        assert!(!ConservativeFlavor::Intel.supports(state, GlPrimitiveTopology::Lines));
        assert!(ConservativeFlavor::Nv.supports(state, GlPrimitiveTopology::Lines));
        state.polygon_mode = GlPolygonMode::Line;
        assert!(!ConservativeFlavor::Intel.supports(state, GlPrimitiveTopology::Triangles));
        state.conservative = false;
        assert!(ConservativeFlavor::Intel.supports(state, GlPrimitiveTopology::Lines));
    }
