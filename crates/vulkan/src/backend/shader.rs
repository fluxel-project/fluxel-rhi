//! Direct SPIR-V shader-module creation.
//!
//! The Vulkan baseline accepts only prevalidated SPIR-V.  It does not pretend
//! that DXIL, WGSL or GLSL is executable without a selected translator.

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
        unsafe { self.device.device.destroy_shader_module(self.module, None) }
    }
}
impl VulkanShader {
    pub(super) fn module(&self) -> vk::ShaderModule {
        self.0.module
    }
}

pub(super) fn create(device: &VulkanDevice, desc: &api::ShaderDesc) -> Result<VulkanShader> {
    if desc.bytes.is_empty() || desc.bytes.len() % 4 != 0 {
        return Err(Error::Shader(
            "SPIR-V binary must contain whole u32 words".into(),
        ));
    }
    let mut bytes = Cursor::new(desc.bytes.as_ref());
    let words =
        read_spv(&mut bytes).map_err(|error| Error::Shader(format!("invalid SPIR-V: {error}")))?;
    let info = vk::ShaderModuleCreateInfo::default().code(&words);
    let module = unsafe { device.0.device.create_shader_module(&info, None) }.map_err(vk_error)?;
    Ok(VulkanShader(Arc::new(ShaderState {
        device: device.0.clone(),
        module,
        entries: desc.entries.clone(),
    })))
}
