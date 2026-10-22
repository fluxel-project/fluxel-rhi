//! Shader module ownership and per-pipeline variants.
//!
//! Raw SPIR-V modules are retained directly. WGSL is validated on creation and
//! compiled for the selected entry point and specialization during pipeline
//! creation. Native variants also adapt push offsets and sampler descriptors;
//! their modules live through the synchronous Vulkan pipeline creation call.

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
    words: Vec<u32>,
    wgsl: Option<api::ShaderDesc>,
}
impl Drop for ShaderState {
    fn drop(&mut self) {
        // SAFETY: ShaderState is the final module owner and retains the live
        // logical device; pipelines retain their modules through creation only.
        if self.module != vk::ShaderModule::null() {
            unsafe { self.device.device.destroy_shader_module(self.module, None) }
        }
    }
}
impl VulkanShader {
    pub(super) fn module(&self) -> vk::ShaderModule {
        self.0.module
    }
    pub(super) fn entries(&self) -> &[api::ShaderEntryPoint] {
        &self.0.entries
    }
    pub(crate) fn words(&self) -> &[u32] {
        &self.0.words
    }
    pub(super) fn mesh_output_mode(&self, entry: &str) -> Option<u32> {
        mesh_output_mode(&self.0.words, entry)
    }
}

// OpEntryPoint names an execution ID; OpExecutionMode declares that ID's
// output topology. Vulkan has no input-assembly topology state for mesh
// pipelines, so the public descriptor must be checked against this mode.
fn mesh_output_mode(words: &[u32], entry: &str) -> Option<u32> {
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return None;
    }
    let mut selected = None;
    let mut offset = 5;
    while offset < words.len() {
        let header = words[offset];
        let count = (header >> 16) as usize;
        if count == 0 || offset.checked_add(count)? > words.len() {
            return None;
        }
        let instruction = &words[offset..offset + count];
        if header & 0xffff == 15 && count >= 4 && instruction[1] == 5365 {
            let name = instruction[3..]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .take_while(|byte| *byte != 0)
                .collect::<Vec<_>>();
            if name == entry.as_bytes() {
                selected = Some(instruction[2]);
            }
        }
        offset += count;
    }
    let id = selected?;
    offset = 5;
    let mut mode = None;
    while offset < words.len() {
        let header = words[offset];
        let count = (header >> 16) as usize;
        if count == 0 || offset.checked_add(count)? > words.len() {
            return None;
        }
        let instruction = &words[offset..offset + count];
        if header & 0xffff == 16
            && count >= 3
            && instruction[1] == id
            && matches!(instruction[2], 27 | 5269 | 5298)
        {
            if mode.replace(instruction[2]).is_some() {
                return None;
            }
        }
        offset += count;
    }
    mode
}

#[cfg(test)]
mod topology_tests {
    use super::*;

    #[test]
    fn reads_the_selected_mesh_entry_topology() {
        let bytes = include_bytes!("../../tests/assets/mesh/mesh.spv");
        let words = read_spv(&mut Cursor::new(bytes.as_slice())).unwrap();
        assert_eq!(mesh_output_mode(&words, "main"), Some(5298));
        assert_eq!(mesh_output_mode(&words, "missing"), None);
    }
}

pub(super) fn create(device: &VulkanDevice, desc: &api::ShaderDesc) -> Result<VulkanShader> {
    for feature in &desc.required_features {
        match feature.as_str() {
            "shaderClipDistance" if device.0.shader_clip_distance => {}
            "shaderClipDistance" => return Err(Error::Unsupported("Vulkan shaderClipDistance")),
            api::SHADER_FEATURE_RAY_QUERY if device.0.ray_support.query => {}
            api::SHADER_FEATURE_RAY_QUERY => return Err(Error::Unsupported("Vulkan ray query")),
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
        words,
        wgsl: None,
    })))
}

#[path = "shader_push.rs"]
mod push;
#[path = "shader_sampler.rs"]
mod sampler;
pub(crate) use sampler::usage as sampler_usage;
#[path = "shader_wgsl.rs"]
mod wgsl;
pub(crate) use wgsl::{compile_variant, create_wgsl};
