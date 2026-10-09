//! Vulkan shader-module ownership.
//!
//! Fluxel's toolchain-facing artifact already fixes the entry point, interface
//! and ABI. Vulkan consumes the SPIR-V words here and keeps only the resulting
//! `VkShaderModule`; pipeline lowering reads the entry-point name and stage from
//! the portable module, so this backend object does not duplicate them.

use std::any::Any;
use std::sync::Arc;

use ash::vk;

use crate::api::error::RhiError;
use crate::api::shader::backend::ShaderModuleBackend;
use crate::api::shader::{ShaderArtifact, ShaderCode};
use crate::backend::vulkan::platform::device::VulkanShared;

#[cfg(test)]
mod tests;

/// One native shader module in the device's shared ownership domain.
pub(crate) struct VulkanShaderModule {
    shared: Arc<VulkanShared>,
    module: vk::ShaderModule,
}

impl VulkanShaderModule {
    pub(crate) fn module(&self) -> vk::ShaderModule {
        self.module
    }
}

impl ShaderModuleBackend for VulkanShaderModule {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for VulkanShaderModule {
    fn drop(&mut self) {
        // SAFETY: the module belongs to `shared.device`; pipeline objects retain
        // their own compiled state, and the portable pipeline keeps the shader
        // handle alive while creation is in progress.
        unsafe { self.shared.device.destroy_shader_module(self.module, None) };
    }
}

/// A failure while preparing a Vulkan shader module.
///
/// Source lowering is already a portable, structured RHI error. Native module
/// creation still reaches the device's loss authority before it is published.
pub(crate) enum ShaderCreationError {
    Lowering(RhiError),
    Native(vk::Result),
}

/// Creates the Vulkan object for one already validated shader artifact.
///
/// SPIR-V is passed directly to Vulkan. With the optional Naga feature, WGSL
/// is lowered to SPIR-V here; no converted code form is exposed in the public
/// artifact or substituted into its stable identity.
pub(crate) fn create_shader(
    shared: Arc<VulkanShared>,
    artifact: &ShaderArtifact,
) -> Result<VulkanShaderModule, ShaderCreationError> {
    let words = match &artifact.code {
        ShaderCode::SpirV(words) => words.clone(),
        _ => crate::api::shader::lowering::naga_spirv(artifact)
            .map_err(ShaderCreationError::Lowering)?,
    };
    let create = vk::ShaderModuleCreateInfo::default().code(&words);
    // SAFETY: `words` is naturally aligned `u32` SPIR-V storage and remains
    // borrowed for the duration of the call. Vulkan copies the module code.
    let module = unsafe { shared.device.create_shader_module(&create, None) }
        .map_err(ShaderCreationError::Native)?;
    Ok(VulkanShaderModule { shared, module })
}
