//! Unit tests for the corresponding implementation module.

use super::*;
#[test]
fn ordinary_and_integer_bindings_reach_native_limit() {
    let limits = vk::PhysicalDeviceLimits {
        max_descriptor_set_samplers: 8,
        max_per_stage_descriptor_samplers: 8,
        ..Default::default()
    };
    let ordinary = sampler_descriptor_count(api::BindingCount::Fixed(8), false);
    assert!(check_sampler_limits(ordinary, &[ordinary], &limits).is_ok());
    let mixed = sampler_descriptor_count(api::BindingCount::Fixed(8), true);
    assert!(check_sampler_limits(mixed, &[mixed], &limits).is_err());
}
