//! Admission checks for direct SPIR-V precision and storage capabilities.
//!
//! Vulkan 1.1/1.2 precision features are opt-in at logical-device creation.
//! A direct artifact may carry these capabilities without a Core feature name,
//! so inspect the compact `OpCapability` declaration list before handing it to
//! the driver. This is deliberately not a full SPIR-V validator.

use super::super::ShaderPrecisionFeatures;
use super::VulkanDevice;
use ash::vk;
use fluxel_rhi_core::api::{Error, Result};

const OP_CAPABILITY: u32 = 17;
const FLOAT16: u32 = 9;
const FLOAT64: u32 = 10;
const INT64: u32 = 11;
const INT16: u32 = 22;
const INT8: u32 = 39;
const STORAGE_BUFFER16: u32 = 4433;
const UNIFORM_AND_STORAGE_BUFFER16: u32 = 4434;
const STORAGE_PUSH_CONSTANT16: u32 = 4435;
const STORAGE_INPUT_OUTPUT16: u32 = 4436;
const STORAGE_BUFFER8: u32 = 4448;
const UNIFORM_AND_STORAGE_BUFFER8: u32 = 4449;
const STORAGE_PUSH_CONSTANT8: u32 = 4450;

pub(super) fn validate_spirv(device: &VulkanDevice, words: &[u32]) -> Result<()> {
    if words.len() < 5 {
        return Err(Error::Shader("invalid SPIR-V header".into()));
    }
    let features = device.0.shader_precision;
    let mut cursor = 5;
    while cursor < words.len() {
        let header = words[cursor];
        let count = (header >> 16) as usize;
        if count == 0
            || cursor
                .checked_add(count)
                .is_none_or(|end| end > words.len())
        {
            return Err(Error::Shader("invalid SPIR-V instruction".into()));
        }
        if header & 0xffff == OP_CAPABILITY && count >= 2 {
            validate_capability(features, device.0.features, words[cursor + 1])?;
        }
        cursor += count;
    }
    Ok(())
}

fn validate_capability(
    features: ShaderPrecisionFeatures,
    base: vk::PhysicalDeviceFeatures,
    capability: u32,
) -> Result<()> {
    match capability {
        FLOAT16 if !features.float16 => Err(Error::Unsupported("Vulkan shaderFloat16")),
        FLOAT64 if base.shader_float64 != vk::TRUE => {
            Err(Error::Unsupported("Vulkan shaderFloat64"))
        }
        INT64 if base.shader_int64 != vk::TRUE => Err(Error::Unsupported("Vulkan shaderInt64")),
        INT16 if base.shader_int16 != vk::TRUE => Err(Error::Unsupported("Vulkan shaderInt16")),
        INT8 if !features.int8 => Err(Error::Unsupported("Vulkan shaderInt8")),
        STORAGE_BUFFER16 if !features.storage_buffer16 => {
            Err(Error::Unsupported("Vulkan storageBuffer16BitAccess"))
        }
        UNIFORM_AND_STORAGE_BUFFER16 if !features.uniform_and_storage_buffer16 => Err(
            Error::Unsupported("Vulkan uniformAndStorageBuffer16BitAccess"),
        ),
        STORAGE_PUSH_CONSTANT16 if !features.storage_push_constant16 => {
            Err(Error::Unsupported("Vulkan storagePushConstant16"))
        }
        STORAGE_INPUT_OUTPUT16 if !features.storage_input_output16 => {
            Err(Error::Unsupported("Vulkan storageInputOutput16"))
        }
        STORAGE_BUFFER8 if !features.storage_buffer8 => {
            Err(Error::Unsupported("Vulkan storageBuffer8BitAccess"))
        }
        UNIFORM_AND_STORAGE_BUFFER8 if !features.uniform_and_storage_buffer8 => Err(
            Error::Unsupported("Vulkan uniformAndStorageBuffer8BitAccess"),
        ),
        STORAGE_PUSH_CONSTANT8 if !features.storage_push_constant8 => {
            Err(Error::Unsupported("Vulkan storagePushConstant8"))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
#[path = "../tests/backend/shader_numeric.rs"]
mod tests;
