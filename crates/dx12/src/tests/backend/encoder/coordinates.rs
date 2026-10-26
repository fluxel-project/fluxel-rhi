use super::*;

fn valid_viewport() -> api::Viewport {
    api::Viewport {
        x: 2.0,
        y: 3.0,
        width: 4.0,
        height: 5.0,
        min_depth: 0.0,
        max_depth: 1.0,
    }
}

#[test]
fn validates_viewport_bounds_before_native_lowering() {
    assert!(viewport(valid_viewport()).is_ok());
    for candidate in [
        api::Viewport {
            width: 0.0,
            ..valid_viewport()
        },
        api::Viewport {
            min_depth: -0.1,
            ..valid_viewport()
        },
        api::Viewport {
            max_depth: 1.1,
            ..valid_viewport()
        },
        api::Viewport {
            min_depth: 0.8,
            max_depth: 0.2,
            ..valid_viewport()
        },
        api::Viewport {
            x: f32::NAN,
            ..valid_viewport()
        },
    ] {
        assert!(matches!(viewport(candidate), Err(Error::InvalidRange)));
    }
}

#[test]
fn validates_scissor_representability_before_native_lowering() {
    assert!(
        scissor(api::Rect {
            x: 2,
            y: 3,
            width: 4,
            height: 5,
        })
        .is_ok()
    );
    for rect in [
        api::Rect {
            x: 0,
            y: 0,
            width: 0,
            height: 1,
        },
        api::Rect {
            x: i32::MAX as u32,
            y: 0,
            width: 1,
            height: 1,
        },
        api::Rect {
            x: u32::MAX,
            y: 0,
            width: 1,
            height: 1,
        },
    ] {
        assert!(matches!(scissor(rect), Err(Error::InvalidRange)));
    }
}
