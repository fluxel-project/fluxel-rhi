//! Logical-device extension dependencies for enabled floating-point atomics.

use crate::backend::instance::shader_atomic_float_extension_required;

#[test]
fn shared_min_max_also_enables_the_base_atomic_float_extension() {
    for route_index in [4, 5] {
        let mut routes = [false; 6];
        routes[route_index] = true;
        assert!(shader_atomic_float_extension_required(routes));
    }
    assert!(!shader_atomic_float_extension_required([false; 6]));
}
