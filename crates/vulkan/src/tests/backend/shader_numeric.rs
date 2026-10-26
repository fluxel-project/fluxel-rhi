//! Unit tests for the corresponding implementation module.

use super::*;

fn enabled() -> ShaderPrecisionFeatures {
    ShaderPrecisionFeatures {
        float16: true,
        int8: true,
        storage_buffer16: true,
        uniform_and_storage_buffer16: true,
        storage_push_constant16: true,
        storage_input_output16: true,
        storage_buffer8: true,
        uniform_and_storage_buffer8: true,
        storage_push_constant8: true,
    }
}

fn enabled_base() -> vk::PhysicalDeviceFeatures {
    vk::PhysicalDeviceFeatures {
        shader_float64: vk::TRUE,
        shader_int64: vk::TRUE,
        shader_int16: vk::TRUE,
        ..Default::default()
    }
}

#[test]
fn maps_each_capability_to_its_enabled_logical_feature() {
    let capabilities = [
        FLOAT16,
        FLOAT64,
        INT64,
        INT16,
        INT8,
        STORAGE_BUFFER16,
        UNIFORM_AND_STORAGE_BUFFER16,
        STORAGE_PUSH_CONSTANT16,
        STORAGE_INPUT_OUTPUT16,
        STORAGE_BUFFER8,
        UNIFORM_AND_STORAGE_BUFFER8,
        STORAGE_PUSH_CONSTANT8,
    ];
    for (index, capability) in capabilities.iter().enumerate() {
        assert!(!capabilities[..index].contains(capability));
        validate_capability(enabled(), enabled_base(), *capability).unwrap();
    }
}

#[test]
fn rejects_float16_when_the_logical_device_did_not_enable_it() {
    let mut features = enabled();
    features.float16 = false;
    assert!(matches!(
        validate_capability(features, enabled_base(), FLOAT16),
        Err(Error::Unsupported("Vulkan shaderFloat16"))
    ));
}

#[test]
fn rejects_unenabled_base_numeric_capabilities() {
    for (capability, expected) in [
        (FLOAT64, "Vulkan shaderFloat64"),
        (INT64, "Vulkan shaderInt64"),
        (INT16, "Vulkan shaderInt16"),
    ] {
        assert!(matches!(
            validate_capability(enabled(), vk::PhysicalDeviceFeatures::default(), capability),
            Err(Error::Unsupported(reason)) if reason == expected
        ));
    }
}
