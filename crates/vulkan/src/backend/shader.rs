//! Direct SPIR-V shader-module creation.
//!
//! The Vulkan baseline accepts only whole-word SPIR-V binaries. Vulkan remains
//! the semantic validation boundary; this module does not pretend that DXIL,
//! WGSL, or GLSL is executable without a selected translator. Shader modules
//! retain their logical device and may be released once pipeline creation ends.

use std::{io::Cursor, sync::Arc};

use ash::{util::read_spv, vk};
use fluxel_rhi_core::api::{self, Error, Result};

use super::{DeviceState, VulkanDevice, vk_error};

#[derive(Clone)]
pub struct VulkanShader(Arc<ShaderState>);
struct ShaderState {
    device: Arc<DeviceState>,
    pub(super) module: vk::ShaderModule,
    pub(super) entries: Vec<api::ShaderEntryPoint>,
}
impl Drop for ShaderState {
    fn drop(&mut self) {
        // SAFETY: ShaderState is the final module owner and retains the live
        // logical device; pipelines retain their modules through creation only.
        unsafe { self.device.device.destroy_shader_module(self.module, None) }
    }
}
impl VulkanShader {
    pub(super) fn module(&self) -> vk::ShaderModule {
        self.0.module
    }
}

pub(super) fn create(device: &VulkanDevice, desc: &api::ShaderDesc) -> Result<VulkanShader> {
    for feature in &desc.required_features {
        match feature.as_str() {
            "shaderClipDistance" if device.0.shader_clip_distance => {}
            "shaderClipDistance" => return Err(Error::Unsupported("Vulkan shaderClipDistance")),
            _ => return Err(Error::Unsupported("Vulkan shader required feature")),
        }
    }
    if desc.bytes.is_empty() || desc.bytes.len() % 4 != 0 {
        return Err(Error::Shader(
            "SPIR-V binary must contain whole u32 words".into(),
        ));
    }
    let mut bytes = Cursor::new(desc.bytes.as_ref());
    let words =
        read_spv(&mut bytes).map_err(|error| Error::Shader(format!("invalid SPIR-V: {error}")))?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    // SAFETY: `words` is a whole-word slice that lives through this call and
    // DeviceState owns the live logical device. Driver-reported SPIR-V semantic
    // errors become `vk_error`; the returned module moves into ShaderState.
    let module = unsafe { device.0.device.create_shader_module(&info, None) }.map_err(vk_error)?;
    Ok(VulkanShader(Arc::new(ShaderState {
        device: device.0.clone(),
        module,
        entries: desc.entries.clone(),
    })))
}
