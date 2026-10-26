    use super::*;

    #[test]
    fn vertex_y_reflection_reverses_native_front_face() {
        let lower = |front_face| {
            lower_primitive(PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face,
                cull_mode: CullMode::None,
                unclipped_depth: false,
                polygon_mode: api::PolygonMode::Fill,
                conservative: false,
            })
            .unwrap()
            .2
        };

        assert_eq!(lower(FrontFace::Ccw), GlFrontFace::Clockwise);
        assert_eq!(lower(FrontFace::Cw), GlFrontFace::CounterClockwise);
    }
