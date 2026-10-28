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
#[path = "../tests/backend/shader/topology_tests.rs"]
mod topology_tests;

#[cfg(test)]
#[path = "../tests/backend/shader/early_depth_tests.rs"]
mod early_depth_tests;

pub(super) fn create(device: &VulkanDevice, desc: &api::ShaderDesc) -> Result<VulkanShader> {
    create_spirv(device, desc, true)
}

/// Creates a temporary native module emitted from a WGSL entry point. Naga
/// already validates WGSL early-depth attributes and emits their SPIR-V
/// execution modes, whose semantics differ from Direct-SPIR-V's explicit
/// `EarlyFragmentTests` feature route.
pub(super) fn create_wgsl_variant(
    device: &VulkanDevice,
    desc: &api::ShaderDesc,
) -> Result<VulkanShader> {
    create_spirv(device, desc, false)
}

fn create_spirv(
    device: &VulkanDevice,
    desc: &api::ShaderDesc,
    validate_direct_early_depth: bool,
) -> Result<VulkanShader> {
    for feature in &desc.required_features {
        match feature.as_str() {
            api::SHADER_FEATURE_MEMORY_COHERENT_SPIRV => {}
            api::SHADER_FEATURE_MEMORY_VOLATILE_SPIRV => {}
            api::SHADER_FEATURE_EARLY_DEPTH_TEST => {}
            api::SHADER_FEATURE_PRIMITIVE_INDEX
                if device.0.features.geometry_shader == vk::TRUE => {}
            api::SHADER_FEATURE_PRIMITIVE_INDEX => {
                return Err(Error::Unsupported("Vulkan primitive index"));
            }
            api::SHADER_FEATURE_DRAW_INDEX if device.0.shader_draw_index => {}
            api::SHADER_FEATURE_DRAW_INDEX => {
                return Err(Error::Unsupported("Vulkan draw index"));
            }
            api::SHADER_FEATURE_BARYCENTRICS
                if device.0.shader_barycentrics_khr || device.0.shader_barycentrics_nv => {}
            api::SHADER_FEATURE_BARYCENTRICS => {
                return Err(Error::Unsupported("Vulkan KHR/NV barycentrics"));
            }
            api::SHADER_FEATURE_BARYCENTRICS_AMD if device.0.shader_barycentrics_amd => {}
            api::SHADER_FEATURE_BARYCENTRICS_AMD => {
                return Err(Error::Unsupported("Vulkan AMD barycentrics"));
            }
            api::SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_ADD
                if device.0.shader_float32_atomic_buffer_add => {}
            api::SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_ADD => {
                return Err(Error::Unsupported("Vulkan Float32 buffer atomic add"));
            }
            api::SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_ADD
                if device.0.shader_float32_atomic_workgroup_add => {}
            api::SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_ADD => {
                return Err(Error::Unsupported("Vulkan Float32 workgroup atomic add"));
            }
            api::SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN
                if device.0.shader_float32_atomic_workgroup_min => {}
            api::SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN => {
                return Err(Error::Unsupported("Vulkan Float32 workgroup atomic min"));
            }
            api::SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX
                if device.0.shader_float32_atomic_workgroup_max => {}
            api::SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX => {
                return Err(Error::Unsupported("Vulkan Float32 workgroup atomic max"));
            }
            api::SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN
                if device.0.shader_float32_atomic_buffer_min => {}
            api::SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN => {
                return Err(Error::Unsupported("Vulkan Float32 buffer atomic min"));
            }
            api::SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX
                if device.0.shader_float32_atomic_buffer_max => {}
            api::SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX => {
                return Err(Error::Unsupported("Vulkan Float32 buffer atomic max"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_ADD
                if device.0.shader_int64_atomic_buffer_add => {}
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_ADD => {
                return Err(Error::Unsupported("Vulkan Int64 buffer atomic add"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN
                if device.0.shader_int64_atomic_buffer_min => {}
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN => {
                return Err(Error::Unsupported("Vulkan Int64 buffer atomic min"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX
                if device.0.shader_int64_atomic_buffer_max => {}
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX => {
                return Err(Error::Unsupported("Vulkan Int64 buffer atomic max"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN
                if device.0.shader_int64_atomic_buffer_umin => {}
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN => {
                return Err(Error::Unsupported(
                    "Vulkan Int64 buffer atomic unsigned min",
                ));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX
                if device.0.shader_int64_atomic_buffer_umax => {}
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX => {
                return Err(Error::Unsupported(
                    "Vulkan Int64 buffer atomic unsigned max",
                ));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR
                if device.0.shader_int64_atomic_buffer_xor => {}
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR => {
                return Err(Error::Unsupported("Vulkan Int64 buffer atomic xor"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE
                if device.0.shader_int64_atomic_buffer_compare_exchange => {}
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE => {
                return Err(Error::Unsupported(
                    "Vulkan Int64 buffer atomic compare-exchange",
                ));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE
                if device.0.shader_int64_atomic_buffer_exchange => {}
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE => {
                return Err(Error::Unsupported("Vulkan Int64 buffer atomic exchange"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND
                if device.0.shader_int64_atomic_buffer_and => {}
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND => {
                return Err(Error::Unsupported("Vulkan Int64 buffer atomic and"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR
                if device.0.shader_int64_atomic_buffer_or => {}
            api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR => {
                return Err(Error::Unsupported("Vulkan Int64 buffer atomic or"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_ADD
                if device.0.shader_int64_atomic_workgroup_add => {}
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_ADD => {
                return Err(Error::Unsupported("Vulkan Int64 workgroup atomic add"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN
                if device.0.shader_int64_atomic_workgroup_min => {}
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN => {
                return Err(Error::Unsupported("Vulkan Int64 workgroup atomic min"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX
                if device.0.shader_int64_atomic_workgroup_max => {}
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX => {
                return Err(Error::Unsupported("Vulkan Int64 workgroup atomic max"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR
                if device.0.shader_int64_atomic_workgroup_xor => {}
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR => {
                return Err(Error::Unsupported("Vulkan Int64 workgroup atomic xor"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND
                if device.0.shader_int64_atomic_workgroup_and => {}
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND => {
                return Err(Error::Unsupported("Vulkan Int64 workgroup atomic and"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR
                if device.0.shader_int64_atomic_workgroup_or => {}
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR => {
                return Err(Error::Unsupported("Vulkan Int64 workgroup atomic or"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN
                if device.0.shader_int64_atomic_workgroup_umin => {}
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN => {
                return Err(Error::Unsupported(
                    "Vulkan Int64 workgroup atomic unsigned min",
                ));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX
                if device.0.shader_int64_atomic_workgroup_umax => {}
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX => {
                return Err(Error::Unsupported(
                    "Vulkan Int64 workgroup atomic unsigned max",
                ));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE
                if device.0.shader_int64_atomic_workgroup_exchange => {}
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE => {
                return Err(Error::Unsupported("Vulkan Int64 workgroup atomic exchange"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE
                if device.0.shader_int64_atomic_workgroup_compare_exchange => {}
            api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE => {
                return Err(Error::Unsupported(
                    "Vulkan Int64 workgroup atomic compare-exchange",
                ));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_ADD
                if device.0.shader_int64_atomic_image_r64uint_add => {}
            api::SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_ADD => {
                return Err(Error::Unsupported("Vulkan R64Uint image atomic add"));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMIN
                if device.0.shader_int64_atomic_image_r64uint_umin => {}
            api::SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMIN => {
                return Err(Error::Unsupported(
                    "Vulkan R64Uint image atomic unsigned min",
                ));
            }
            api::SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMAX
                if device.0.shader_int64_atomic_image_r64uint_umax => {}
            api::SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMAX => {
                return Err(Error::Unsupported(
                    "Vulkan R64Uint image atomic unsigned max",
                ));
            }
            api::SHADER_FEATURE_EXPERIMENTAL_COOPERATIVE_MATRIX
                if !device.0.cooperative_matrix_configurations.is_empty() => {}
            api::SHADER_FEATURE_EXPERIMENTAL_COOPERATIVE_MATRIX => {
                return Err(Error::Unsupported("Vulkan cooperative matrix"));
            }
            "shaderClipDistance" if device.0.shader_clip_distance => {}
            "shaderClipDistance" => return Err(Error::Unsupported("Vulkan shaderClipDistance")),
            api::SHADER_FEATURE_RAY_QUERY if device.0.ray_support.query => {}
            api::SHADER_FEATURE_RAY_QUERY => return Err(Error::Unsupported("Vulkan ray query")),
            "subgroups" => {}
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
    validate_barycentric_route(device, desc, &words)?;
    if validate_direct_early_depth {
        validate_early_depth_route(desc, &words)?;
    }
    validate_float32_atomic_route(device, desc, &words)?;
    validate_int64_atomic_route(device, desc, &words)?;
    super::shader_memory_decorations::validate(desc, &words)?;
    validate_cooperative_matrix_route(device, desc, &words)?;
    subgroup::validate_spirv(device, desc, &words)?;
    numeric::validate_spirv(device, &words)?;
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

/// Validates Direct SPIR-V admission for the public early-depth requirement.
///
/// WGSL is validated by Naga before this path. A native artifact is opaque to
/// Core, so the requirement and every declared fragment entry's
/// `EarlyFragmentTests` mode must correspond before a native shader module is
/// created.
fn validate_early_depth_route(desc: &api::ShaderDesc, words: &[u32]) -> Result<()> {
    const OP_ENTRY_POINT: u32 = 15;
    const OP_EXECUTION_MODE: u32 = 16;
    const EXECUTION_MODEL_FRAGMENT: u32 = 4;
    const EXECUTION_MODE_EARLY_FRAGMENT_TESTS: u32 = 9;

    let declared = desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_EARLY_DEPTH_TEST);
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(Error::Shader("invalid SPIR-V header".into()));
    }

    let mut entries = std::collections::HashMap::<String, u32>::new();
    let mut fragment_entry_ids = std::collections::HashSet::<u32>::new();
    let mut early_fragment_tests = std::collections::HashSet::<u32>::new();
    let mut offset = 5usize;
    while offset < words.len() {
        let header = words[offset];
        let count = (header >> 16) as usize;
        let opcode = header & 0xffff;
        let end = offset
            .checked_add(count)
            .filter(|end| count != 0 && *end <= words.len())
            .ok_or_else(|| Error::Shader("invalid SPIR-V instruction length".into()))?;
        let instruction = &words[offset..end];
        match opcode {
            OP_ENTRY_POINT if count >= 4 && instruction[1] == EXECUTION_MODEL_FRAGMENT => {
                let name = instruction[3..]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .take_while(|byte| *byte != 0)
                    .collect::<Vec<_>>();
                let name = std::str::from_utf8(&name)
                    .map_err(|_| Error::Shader("invalid SPIR-V entry-point name".into()))?
                    .to_owned();
                if entries.insert(name, instruction[2]).is_some() {
                    return Err(Error::Shader(
                        "duplicate SPIR-V fragment entry point".into(),
                    ));
                }
                fragment_entry_ids.insert(instruction[2]);
            }
            OP_EXECUTION_MODE
                if count >= 3 && instruction[2] == EXECUTION_MODE_EARLY_FRAGMENT_TESTS =>
            {
                early_fragment_tests.insert(instruction[1]);
            }
            _ => {}
        }
        offset = end;
    }

    if !declared
        && fragment_entry_ids
            .iter()
            .any(|id| early_fragment_tests.contains(id))
    {
        return Err(Error::Shader(
            "SPIR-V EarlyFragmentTests requires shader-early-depth-test feature".into(),
        ));
    }

    for entry in desc
        .entries
        .iter()
        .filter(|entry| entry.stage == api::ShaderStage::Fragment)
    {
        let id = entries.get(&entry.name).copied();
        if declared && id.is_none() {
            return Err(Error::Unsupported(
                "Vulkan early-depth SPIR-V fragment entry point is missing",
            ));
        }
        let uses_early_fragment_tests = id.is_some_and(|id| early_fragment_tests.contains(&id));
        if declared && !uses_early_fragment_tests {
            return Err(Error::Unsupported(
                "Vulkan early-depth SPIR-V requires EarlyFragmentTests execution mode",
            ));
        }
        if !declared && uses_early_fragment_tests {
            return Err(Error::Shader(
                "SPIR-V EarlyFragmentTests requires shader-early-depth-test feature".into(),
            ));
        }
    }
    Ok(())
}

fn validate_float32_atomic_route(
    device: &VulkanDevice,
    desc: &api::ShaderDesc,
    words: &[u32],
) -> Result<()> {
    const ATOMIC_FLOAT32_ADD_EXT_CAPABILITY: u32 = 6033;
    const OP_ATOMIC_F_ADD_EXT: u32 = 6035;
    const ATOMIC_FLOAT32_MIN_MAX_EXT_CAPABILITY: u32 = 5612;
    const OP_ATOMIC_F_MIN_EXT: u32 = 5614;
    const OP_ATOMIC_F_MAX_EXT: u32 = 5615;
    const OP_TYPE_FLOAT: u32 = 22;
    const OP_TYPE_POINTER: u32 = 32;
    const OP_VARIABLE: u32 = 59;
    const OP_ACCESS_CHAIN: u32 = 65;
    const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;
    const OP_PTR_ACCESS_CHAIN: u32 = 67;
    const OP_IN_BOUNDS_PTR_ACCESS_CHAIN: u32 = 70;
    const OP_COPY_OBJECT: u32 = 83;
    let declared_add = desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_ADD);
    let declared_workgroup_add = desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_ADD);
    let declared_workgroup_min = desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MIN);
    let declared_workgroup_max = desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_FLOAT32_ATOMIC_WORKGROUP_MAX);
    let declared_min = desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MIN);
    let declared_max = desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_FLOAT32_ATOMIC_BUFFER_MAX);
    if (declared_add as u8
        + declared_workgroup_add as u8
        + declared_min as u8
        + declared_workgroup_min as u8
        + declared_max as u8
        + declared_workgroup_max as u8)
        > 1
    {
        return Err(Error::Unsupported(
            "conflicting Vulkan Float32 atomic feature routes",
        ));
    }
    let declared = declared_add
        || declared_workgroup_add
        || declared_min
        || declared_workgroup_min
        || declared_max
        || declared_workgroup_max;
    let (
        expected_capability,
        expected_opcode,
        expected_extension,
        device_supported,
        name,
        expected_storage_class,
    ) = if declared_min {
        (
            ATOMIC_FLOAT32_MIN_MAX_EXT_CAPABILITY,
            OP_ATOMIC_F_MIN_EXT,
            b"SPV_EXT_shader_atomic_float_min_max".as_slice(),
            device.0.shader_float32_atomic_buffer_min,
            "min",
            12,
        )
    } else if declared_workgroup_min {
        (
            ATOMIC_FLOAT32_MIN_MAX_EXT_CAPABILITY,
            OP_ATOMIC_F_MIN_EXT,
            b"SPV_EXT_shader_atomic_float_min_max".as_slice(),
            device.0.shader_float32_atomic_workgroup_min,
            "workgroup min",
            4,
        )
    } else if declared_max {
        (
            ATOMIC_FLOAT32_MIN_MAX_EXT_CAPABILITY,
            OP_ATOMIC_F_MAX_EXT,
            b"SPV_EXT_shader_atomic_float_min_max".as_slice(),
            device.0.shader_float32_atomic_buffer_max,
            "max",
            12,
        )
    } else if declared_workgroup_max {
        (
            ATOMIC_FLOAT32_MIN_MAX_EXT_CAPABILITY,
            OP_ATOMIC_F_MAX_EXT,
            b"SPV_EXT_shader_atomic_float_min_max".as_slice(),
            device.0.shader_float32_atomic_workgroup_max,
            "workgroup max",
            4,
        )
    } else if declared_workgroup_add {
        (
            ATOMIC_FLOAT32_ADD_EXT_CAPABILITY,
            OP_ATOMIC_F_ADD_EXT,
            b"SPV_EXT_shader_atomic_float_add".as_slice(),
            device.0.shader_float32_atomic_workgroup_add,
            "workgroup add",
            4,
        )
    } else {
        (
            ATOMIC_FLOAT32_ADD_EXT_CAPABILITY,
            OP_ATOMIC_F_ADD_EXT,
            b"SPV_EXT_shader_atomic_float_add".as_slice(),
            device.0.shader_float32_atomic_buffer_add,
            "add",
            12,
        )
    };
    let mut extension = false;
    let mut capability = false;
    let mut operation = false;
    let mut any_float_atomic = false;
    let mut operation_outside_route = false;
    let mut atomic_pointers_match_float32_route = true;
    let mut float32_type_ids = std::collections::HashSet::<u32>::new();
    let mut pointer_types = std::collections::HashMap::<u32, (u32, u32)>::new();
    let mut pointer_value_types = std::collections::HashMap::<u32, u32>::new();
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(Error::Shader("invalid SPIR-V header".into()));
    }
    let mut offset = 5usize;
    while offset < words.len() {
        let header = words[offset];
        let count = (header >> 16) as usize;
        let opcode = header & 0xffff;
        let end = offset
            .checked_add(count)
            .filter(|end| count != 0 && *end <= words.len())
            .ok_or_else(|| Error::Shader("invalid SPIR-V instruction length".into()))?;
        match opcode {
            OP_TYPE_FLOAT if count >= 3 && words[offset + 2] == 32 => {
                float32_type_ids.insert(words[offset + 1]);
            }
            OP_TYPE_POINTER if count >= 4 => {
                pointer_types.insert(words[offset + 1], (words[offset + 2], words[offset + 3]));
            }
            OP_VARIABLE if count >= 4 => {
                pointer_value_types.insert(words[offset + 2], words[offset + 1]);
            }
            OP_ACCESS_CHAIN
            | OP_IN_BOUNDS_ACCESS_CHAIN
            | OP_PTR_ACCESS_CHAIN
            | OP_IN_BOUNDS_PTR_ACCESS_CHAIN
            | OP_COPY_OBJECT
                if count >= 4 =>
            {
                pointer_value_types.insert(words[offset + 2], words[offset + 1]);
            }
            10 if count >= 2 => {
                let name = words[offset + 1..end]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .take_while(|byte| *byte != 0)
                    .collect::<Vec<_>>();
                extension |= name == expected_extension;
            }
            17 if count >= 2 => capability |= words[offset + 1] == expected_capability,
            OP_ATOMIC_F_ADD_EXT | OP_ATOMIC_F_MIN_EXT | OP_ATOMIC_F_MAX_EXT if count >= 7 => {
                any_float_atomic = true;
                if opcode != expected_opcode {
                    operation_outside_route = true;
                } else {
                    operation = true;
                    let pointer_type = pointer_value_types
                        .get(&words[offset + 3])
                        .and_then(|type_id| pointer_types.get(type_id));
                    atomic_pointers_match_float32_route &= float32_atomic_pointer_matches(
                        pointer_type,
                        expected_storage_class,
                        &float32_type_ids,
                    );
                }
            }
            _ => {}
        }
        offset = end;
    }
    if any_float_atomic && !declared {
        return Err(Error::Shader(
            "Float32 atomic instructions require a declared atomic feature route".into(),
        ));
    }
    if operation_outside_route {
        return Err(Error::Unsupported(
            "Vulkan Float32 atomic operation is outside the declared route",
        ));
    }
    if operation && (!extension || !capability) {
        return Err(Error::Shader(format!(
            "Float32 atomic-{name} requires its declared route and SPIR-V extension/capability"
        )));
    }
    if declared && (!operation || !extension || !capability || !atomic_pointers_match_float32_route)
    {
        return Err(Error::Shader(format!(
            "Float32 {name} route requires its matching 32-bit float pointer, storage class, SPIR-V operation, extension and capability"
        )));
    }
    if declared && !device_supported {
        return Err(Error::Unsupported("Vulkan Float32 atomic operation"));
    }
    Ok(())
}

fn validate_int64_atomic_route(
    device: &VulkanDevice,
    desc: &api::ShaderDesc,
    words: &[u32],
) -> Result<()> {
    const INT64_ATOMICS_CAPABILITY: u32 = 12;
    const INT64_IMAGE_EXT_CAPABILITY: u32 = 5016;
    const OP_EXTENSION: u32 = 10;
    const OP_TYPE_INT: u32 = 21;
    const OP_TYPE_POINTER: u32 = 32;
    const OP_VARIABLE: u32 = 59;
    const OP_ACCESS_CHAIN: u32 = 65;
    const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;
    const OP_PTR_ACCESS_CHAIN: u32 = 67;
    const OP_IN_BOUNDS_PTR_ACCESS_CHAIN: u32 = 70;
    const OP_COPY_OBJECT: u32 = 83;
    const OP_TYPE_IMAGE: u32 = 25;
    const OP_IMAGE_TEXEL_POINTER: u32 = 60;
    const OP_CAPABILITY: u32 = 17;
    const OP_ATOMIC_LOAD: u32 = 227;
    const OP_ATOMIC_STORE: u32 = 228;
    const OP_ATOMIC_EXCHANGE: u32 = 229;
    const OP_ATOMIC_IADD: u32 = 234;
    const OP_ATOMIC_SMIN: u32 = 236;
    const OP_ATOMIC_SMAX: u32 = 238;
    const OP_ATOMIC_UMIN: u32 = 237;
    const OP_ATOMIC_UMAX: u32 = 239;
    const OP_ATOMIC_XOR: u32 = 242;
    const OP_ATOMIC_AND: u32 = 240;
    const OP_ATOMIC_OR: u32 = 241;
    const OP_ATOMIC_COMPARE_EXCHANGE: u32 = 230;
    const FIRST_ATOMIC_RESULT_OP: u32 = 229;
    const LAST_ATOMIC_RESULT_OP: u32 = 242;
    const STORAGE_CLASS_IMAGE: u32 = 11;
    const IMAGE_FORMAT_R64UI: u32 = 40;

    let declared_features = desc
        .required_features
        .iter()
        .filter(|feature| {
            feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_ADD
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_ADD
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_ADD
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMIN
                || feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMAX
        })
        .collect::<Vec<_>>();
    if declared_features.len() > 1 {
        return Err(Error::Unsupported(
            "conflicting Vulkan Int64 atomic feature routes",
        ));
    }
    let (expected_opcode, device_supported, operation_name, expected_storage_class) =
        match declared_features.first() {
            Some(feature)
                if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_ADD =>
            {
                (
                    OP_ATOMIC_IADD,
                    device.0.shader_int64_atomic_image_r64uint_add,
                    "R64Uint image add",
                    STORAGE_CLASS_IMAGE,
                )
            }
            Some(feature)
                if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMIN =>
            {
                (
                    OP_ATOMIC_UMIN,
                    device.0.shader_int64_atomic_image_r64uint_umin,
                    "R64Uint image unsigned min",
                    STORAGE_CLASS_IMAGE,
                )
            }
            Some(feature)
                if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_IMAGE_R64UINT_UMAX =>
            {
                (
                    OP_ATOMIC_UMAX,
                    device.0.shader_int64_atomic_image_r64uint_umax,
                    "R64Uint image unsigned max",
                    STORAGE_CLASS_IMAGE,
                )
            }
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_MIN => (
                OP_ATOMIC_SMIN,
                device.0.shader_int64_atomic_buffer_min,
                "signed min",
                12,
            ),
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_MAX => (
                OP_ATOMIC_SMAX,
                device.0.shader_int64_atomic_buffer_max,
                "signed max",
                12,
            ),
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMIN => (
                OP_ATOMIC_UMIN,
                device.0.shader_int64_atomic_buffer_umin,
                "unsigned min",
                12,
            ),
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_UMAX => (
                OP_ATOMIC_UMAX,
                device.0.shader_int64_atomic_buffer_umax,
                "unsigned max",
                12,
            ),
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_XOR => (
                OP_ATOMIC_XOR,
                device.0.shader_int64_atomic_buffer_xor,
                "bitwise xor",
                12,
            ),
            Some(feature)
                if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_COMPARE_EXCHANGE =>
            {
                (
                    OP_ATOMIC_COMPARE_EXCHANGE,
                    device.0.shader_int64_atomic_buffer_compare_exchange,
                    "compare-exchange",
                    12,
                )
            }
            Some(feature)
                if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_EXCHANGE =>
            {
                (
                    OP_ATOMIC_EXCHANGE,
                    device.0.shader_int64_atomic_buffer_exchange,
                    "exchange",
                    12,
                )
            }
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_AND => (
                OP_ATOMIC_AND,
                device.0.shader_int64_atomic_buffer_and,
                "bitwise and",
                12,
            ),
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_BUFFER_OR => (
                OP_ATOMIC_OR,
                device.0.shader_int64_atomic_buffer_or,
                "bitwise or",
                12,
            ),
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MIN => {
                (
                    OP_ATOMIC_SMIN,
                    device.0.shader_int64_atomic_workgroup_min,
                    "workgroup signed min",
                    4,
                )
            }
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_MAX => {
                (
                    OP_ATOMIC_SMAX,
                    device.0.shader_int64_atomic_workgroup_max,
                    "workgroup signed max",
                    4,
                )
            }
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_ADD => {
                (
                    OP_ATOMIC_IADD,
                    device.0.shader_int64_atomic_workgroup_add,
                    "workgroup add",
                    4,
                )
            }
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_XOR => {
                (
                    OP_ATOMIC_XOR,
                    device.0.shader_int64_atomic_workgroup_xor,
                    "workgroup bitwise xor",
                    4,
                )
            }
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_AND => {
                (
                    OP_ATOMIC_AND,
                    device.0.shader_int64_atomic_workgroup_and,
                    "workgroup bitwise and",
                    4,
                )
            }
            Some(feature) if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_OR => (
                OP_ATOMIC_OR,
                device.0.shader_int64_atomic_workgroup_or,
                "workgroup bitwise or",
                4,
            ),
            Some(feature)
                if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMIN =>
            {
                (
                    OP_ATOMIC_UMIN,
                    device.0.shader_int64_atomic_workgroup_umin,
                    "workgroup unsigned min",
                    4,
                )
            }
            Some(feature)
                if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_UMAX =>
            {
                (
                    OP_ATOMIC_UMAX,
                    device.0.shader_int64_atomic_workgroup_umax,
                    "workgroup unsigned max",
                    4,
                )
            }
            Some(feature)
                if feature.as_str() == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_EXCHANGE =>
            {
                (
                    OP_ATOMIC_EXCHANGE,
                    device.0.shader_int64_atomic_workgroup_exchange,
                    "workgroup exchange",
                    4,
                )
            }
            Some(feature)
                if feature.as_str()
                    == api::SHADER_FEATURE_INT64_ATOMIC_WORKGROUP_COMPARE_EXCHANGE =>
            {
                (
                    OP_ATOMIC_COMPARE_EXCHANGE,
                    device.0.shader_int64_atomic_workgroup_compare_exchange,
                    "workgroup compare-exchange",
                    4,
                )
            }
            Some(_) => (
                OP_ATOMIC_IADD,
                device.0.shader_int64_atomic_buffer_add,
                "add",
                12,
            ),
            None => (OP_ATOMIC_IADD, false, "declared operation", 12),
        };

    let mut declares_int64_atomics = false;
    let mut declares_int64_image = false;
    let mut declares_int64_image_extension = false;
    let mut int64_type_ids = std::collections::HashSet::<u32>::new();
    let mut signed_int64_type_ids = std::collections::HashSet::<u32>::new();
    let mut unsigned_int64_type_ids = std::collections::HashSet::<u32>::new();
    let mut pointer_types = std::collections::HashMap::<u32, (u32, u32)>::new();
    let mut pointer_value_types = std::collections::HashMap::<u32, u32>::new();
    let mut image_types = std::collections::HashMap::<u32, (u32, u32, u32, u32, u32, u32)>::new();
    let mut image_variables = std::collections::HashSet::<u32>::new();
    let mut expected_image_pointer_values = std::collections::HashSet::<u32>::new();
    let mut saw_expected_atomic = false;
    let mut saw_int64_atomic = false;
    let mut expected_atomic_uses_storage_class = true;
    let mut saw_other_int64_atomic = false;

    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(Error::Shader("invalid SPIR-V header".into()));
    }
    let mut offset = 5usize;
    while offset < words.len() {
        let header = words[offset];
        let count = (header >> 16) as usize;
        let opcode = header & 0xffff;
        let end = offset
            .checked_add(count)
            .filter(|end| count != 0 && *end <= words.len())
            .ok_or_else(|| Error::Shader("invalid SPIR-V instruction length".into()))?;
        match opcode {
            OP_CAPABILITY if count >= 2 => {
                declares_int64_atomics |= words[offset + 1] == INT64_ATOMICS_CAPABILITY;
                declares_int64_image |= words[offset + 1] == INT64_IMAGE_EXT_CAPABILITY;
            }
            OP_EXTENSION if count >= 2 => {
                let extension = words[offset + 1..end]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .take_while(|byte| *byte != 0)
                    .collect::<Vec<_>>();
                declares_int64_image_extension |= extension == b"SPV_EXT_shader_image_int64";
            }
            OP_TYPE_INT if count >= 4 && words[offset + 2] == 64 => {
                int64_type_ids.insert(words[offset + 1]);
                if words[offset + 3] == 1 {
                    signed_int64_type_ids.insert(words[offset + 1]);
                } else {
                    unsigned_int64_type_ids.insert(words[offset + 1]);
                }
            }
            OP_TYPE_POINTER if count >= 4 => {
                pointer_types.insert(words[offset + 1], (words[offset + 2], words[offset + 3]));
            }
            OP_TYPE_IMAGE if count >= 9 => {
                image_types.insert(
                    words[offset + 1],
                    (
                        words[offset + 2],
                        words[offset + 3],
                        words[offset + 5],
                        words[offset + 6],
                        words[offset + 7],
                        words[offset + 8],
                    ),
                );
            }
            OP_VARIABLE if count >= 4 => {
                pointer_value_types.insert(words[offset + 2], words[offset + 1]);
                if let Some((storage_class, pointee)) = pointer_types.get(&words[offset + 1]) {
                    if *storage_class == 0
                        && image_types.get(pointee).is_some_and(
                            |(sampled_type, dim, arrayed, multisampled, sampled, format)| {
                                unsigned_int64_type_ids.contains(sampled_type)
                                    && *dim == 1
                                    && *arrayed == 0
                                    && *multisampled == 0
                                    && *sampled == 2
                                    && *format == IMAGE_FORMAT_R64UI
                            },
                        )
                    {
                        image_variables.insert(words[offset + 2]);
                    }
                }
            }
            OP_IMAGE_TEXEL_POINTER if count >= 6 => {
                let result_type = words[offset + 1];
                let result_id = words[offset + 2];
                let image_id = words[offset + 3];
                pointer_value_types.insert(result_id, result_type);
                if image_variables.contains(&image_id)
                    && pointer_types
                        .get(&result_type)
                        .is_some_and(|(storage_class, pointee)| {
                            *storage_class == STORAGE_CLASS_IMAGE
                                && unsigned_int64_type_ids.contains(pointee)
                        })
                {
                    expected_image_pointer_values.insert(result_id);
                }
            }
            OP_ACCESS_CHAIN
            | OP_IN_BOUNDS_ACCESS_CHAIN
            | OP_PTR_ACCESS_CHAIN
            | OP_IN_BOUNDS_PTR_ACCESS_CHAIN
            | OP_COPY_OBJECT
                if count >= 4 =>
            {
                pointer_value_types.insert(words[offset + 2], words[offset + 1]);
            }
            OP_ATOMIC_LOAD | OP_ATOMIC_STORE | FIRST_ATOMIC_RESULT_OP..=LAST_ATOMIC_RESULT_OP => {
                let is_store = opcode == OP_ATOMIC_STORE;
                let pointer_id = if is_store {
                    (count >= 4).then(|| words[offset + 1])
                } else {
                    (count >= 5).then(|| words[offset + 3])
                };
                if let Some(pointer_id) = pointer_id {
                    let pointer_type_id = pointer_value_types.get(&pointer_id).copied();
                    let pointer_type = pointer_type_id.and_then(|id| pointer_types.get(&id));
                    let uses_int64 =
                        pointer_type.is_some_and(|(_, pointee)| int64_type_ids.contains(pointee));
                    if !uses_int64 && !is_store && count >= 3 {
                        // Atomic result type catches integer widths when the
                        // pointer provenance came through an unsupported form.
                        if int64_type_ids.contains(&words[offset + 1]) {
                            saw_int64_atomic = true;
                            if opcode == expected_opcode {
                                saw_expected_atomic = true;
                                expected_atomic_uses_storage_class = false;
                            } else {
                                saw_other_int64_atomic = true;
                            }
                        }
                    } else if uses_int64 {
                        saw_int64_atomic = true;
                        if opcode == expected_opcode {
                            saw_expected_atomic = true;
                            expected_atomic_uses_storage_class &= int64_atomic_pointer_matches(
                                pointer_type,
                                expected_storage_class,
                                &int64_type_ids,
                                &signed_int64_type_ids,
                                &unsigned_int64_type_ids,
                                matches!(expected_opcode, OP_ATOMIC_SMIN | OP_ATOMIC_SMAX),
                                matches!(expected_opcode, OP_ATOMIC_UMIN | OP_ATOMIC_UMAX),
                            );
                            if expected_storage_class == STORAGE_CLASS_IMAGE {
                                expected_atomic_uses_storage_class &=
                                    expected_image_pointer_values.contains(&pointer_id);
                            }
                        } else {
                            saw_other_int64_atomic = true;
                        }
                    }
                }
            }
            _ => {}
        }
        offset = end;
    }

    let declared = !declared_features.is_empty();
    if declares_int64_atomics && !declared {
        return Err(Error::Shader(
            "Int64Atomics SPIR-V capability requires a declared atomic feature route".into(),
        ));
    }
    if declares_int64_image && !declared {
        return Err(Error::Shader(
            "Int64ImageEXT SPIR-V capability requires a declared image atomic feature route".into(),
        ));
    }
    if saw_int64_atomic && !declared {
        return Err(Error::Shader(
            "Int64 atomic instructions require a declared atomic feature route".into(),
        ));
    }
    if saw_other_int64_atomic {
        return Err(Error::Unsupported(
            "Vulkan Int64 atomic operation is outside the declared route",
        ));
    }
    if saw_expected_atomic && (!declared || !declares_int64_atomics) {
        return Err(Error::Shader(
            "Int64 atomic instruction requires its declared route and Int64Atomics SPIR-V capability".into(),
        ));
    }
    if declared
        && (!saw_expected_atomic
            || !declares_int64_atomics
            || !expected_atomic_uses_storage_class
            || (expected_storage_class == STORAGE_CLASS_IMAGE
                && (!declares_int64_image || !declares_int64_image_extension)))
    {
        return Err(Error::Shader(format!(
            "Int64 {operation_name} route requires the matching SPIR-V operation and storage class with Int64Atomics capability"
        )));
    }
    if declared && !device_supported {
        return Err(Error::Unsupported("Vulkan Int64 atomic operation"));
    }
    Ok(())
}

fn float32_atomic_pointer_matches(
    pointer_type: Option<&(u32, u32)>,
    expected_storage_class: u32,
    float32_type_ids: &std::collections::HashSet<u32>,
) -> bool {
    pointer_type.is_some_and(|(storage_class, pointee_type)| {
        *storage_class == expected_storage_class && float32_type_ids.contains(pointee_type)
    })
}

fn int64_atomic_pointer_matches(
    pointer_type: Option<&(u32, u32)>,
    expected_storage_class: u32,
    int64_type_ids: &std::collections::HashSet<u32>,
    signed_int64_type_ids: &std::collections::HashSet<u32>,
    unsigned_int64_type_ids: &std::collections::HashSet<u32>,
    require_signed: bool,
    require_unsigned: bool,
) -> bool {
    pointer_type.is_some_and(|(storage_class, pointee_type)| {
        *storage_class == expected_storage_class
            && int64_type_ids.contains(pointee_type)
            && (!require_signed || signed_int64_type_ids.contains(pointee_type))
            && (!require_unsigned || unsigned_int64_type_ids.contains(pointee_type))
    })
}

#[cfg(test)]
#[path = "../tests/backend/shader_atomic_type.rs"]
mod atomic_type_tests;
#[cfg(test)]
#[path = "../tests/backend/shader/cooperative_matrix_tests.rs"]
mod cooperative_matrix_tests;

/// Checks that native cooperative-matrix SPIR-V is explicitly admitted for
/// one of the exact tuples reported when this logical device was opened.
///
/// The extension is deliberately not inferred from an arbitrary SPIR-V module:
/// callers must opt into the experimental route, and the module must contain a
/// complete load/multiply-add/store operation for the selected native tuple.
fn validate_cooperative_matrix_route(
    device: &VulkanDevice,
    desc: &api::ShaderDesc,
    words: &[u32],
) -> Result<()> {
    const COOPERATIVE_MATRIX_KHR_CAPABILITY: u32 = 6022;
    const OP_EXTENSION: u32 = 10;
    const OP_CAPABILITY: u32 = 17;
    const OP_TYPE_INT: u32 = 21;
    const OP_TYPE_FLOAT: u32 = 22;
    const OP_CONSTANT: u32 = 43;
    const OP_CONSTANT_COMPOSITE: u32 = 44;
    const OP_LOAD: u32 = 61;
    const OP_COPY_OBJECT: u32 = 83;
    const OP_TYPE_COOPERATIVE_MATRIX_KHR: u32 = 4456;
    const OP_COOPERATIVE_MATRIX_LOAD_KHR: u32 = 4457;
    const OP_COOPERATIVE_MATRIX_STORE_KHR: u32 = 4458;
    const OP_COOPERATIVE_MATRIX_MUL_ADD_KHR: u32 = 4459;
    const USE_A: u32 = 0;
    const USE_B: u32 = 1;
    const USE_ACCUMULATOR: u32 = 2;
    const SCOPE_DEVICE: u32 = 1;
    const SCOPE_WORKGROUP: u32 = 2;
    const SCOPE_SUBGROUP: u32 = 3;
    const SCOPE_QUEUE_FAMILY: u32 = 5;

    #[derive(Clone, Copy)]
    struct MatrixType {
        component: api::CooperativeMatrixComponentType,
        scope: api::CooperativeMatrixScope,
        rows: u32,
        columns: u32,
        use_kind: u32,
    }

    let declared = desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_EXPERIMENTAL_COOPERATIVE_MATRIX);
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(Error::Shader("invalid SPIR-V header".into()));
    }
    validate_cooperative_matrix_layout(words)?;

    let mut extension = false;
    let mut capability = false;
    let mut scalar_types =
        std::collections::HashMap::<u32, api::CooperativeMatrixComponentType>::new();
    let mut constants = std::collections::HashMap::<u32, u32>::new();
    let mut matrices = std::collections::HashMap::<u32, MatrixType>::new();
    let mut saw_matrix_type = false;
    let mut value_types = std::collections::HashMap::<u32, u32>::new();
    let mut loads = std::collections::HashSet::<u32>::new();
    let mut stores = std::collections::HashSet::<u32>::new();
    let mut mul_adds = Vec::<(u32, u32, u32, u32)>::new();

    let mut offset = 5usize;
    while offset < words.len() {
        let header = words[offset];
        let count = (header >> 16) as usize;
        let opcode = header & 0xffff;
        let end = offset
            .checked_add(count)
            .filter(|end| count != 0 && *end <= words.len())
            .ok_or_else(|| Error::Shader("invalid SPIR-V instruction length".into()))?;
        match opcode {
            OP_EXTENSION if count >= 2 => {
                let name = words[offset + 1..end]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .take_while(|byte| *byte != 0)
                    .collect::<Vec<_>>();
                extension |= name == b"SPV_KHR_cooperative_matrix";
            }
            OP_CAPABILITY if count >= 2 => {
                capability |= words[offset + 1] == COOPERATIVE_MATRIX_KHR_CAPABILITY;
            }
            OP_TYPE_FLOAT if count >= 3 => {
                let component = match words[offset + 2] {
                    16 => Some(api::CooperativeMatrixComponentType::Float16),
                    32 => Some(api::CooperativeMatrixComponentType::Float32),
                    64 => Some(api::CooperativeMatrixComponentType::Float64),
                    _ => None,
                };
                if let Some(component) = component {
                    scalar_types.insert(words[offset + 1], component);
                }
            }
            OP_TYPE_INT if count >= 4 => {
                let component = match (words[offset + 2], words[offset + 3]) {
                    (8, 1) => Some(api::CooperativeMatrixComponentType::Sint8),
                    (16, 1) => Some(api::CooperativeMatrixComponentType::Sint16),
                    (32, 1) => Some(api::CooperativeMatrixComponentType::Sint32),
                    (64, 1) => Some(api::CooperativeMatrixComponentType::Sint64),
                    (8, 0) => Some(api::CooperativeMatrixComponentType::Uint8),
                    (16, 0) => Some(api::CooperativeMatrixComponentType::Uint16),
                    (32, 0) => Some(api::CooperativeMatrixComponentType::Uint32),
                    (64, 0) => Some(api::CooperativeMatrixComponentType::Uint64),
                    _ => None,
                };
                if let Some(component) = component {
                    scalar_types.insert(words[offset + 1], component);
                }
            }
            OP_CONSTANT if count >= 4 => {
                // Matrix dimensions and scopes are 32-bit scalar constants in
                // the KHR type declaration. Wider constants cannot describe a
                // legal matrix tuple and are intentionally ignored here.
                constants.insert(words[offset + 2], words[offset + 3]);
            }
            OP_CONSTANT_COMPOSITE | OP_LOAD | OP_COPY_OBJECT if count >= 3 => {
                value_types.insert(words[offset + 2], words[offset + 1]);
            }
            OP_TYPE_COOPERATIVE_MATRIX_KHR if count == 7 => {
                saw_matrix_type = true;
                let component = scalar_types.get(&words[offset + 2]).copied();
                let scope = match constants.get(&words[offset + 3]).copied() {
                    Some(SCOPE_SUBGROUP) => Some(api::CooperativeMatrixScope::Subgroup),
                    Some(SCOPE_WORKGROUP) => Some(api::CooperativeMatrixScope::Workgroup),
                    Some(SCOPE_DEVICE) => Some(api::CooperativeMatrixScope::Device),
                    Some(SCOPE_QUEUE_FAMILY) => Some(api::CooperativeMatrixScope::QueueFamily),
                    _ => None,
                };
                let rows = constants.get(&words[offset + 4]).copied();
                let columns = constants.get(&words[offset + 5]).copied();
                if let (Some(component), Some(scope), Some(rows), Some(columns)) =
                    (component, scope, rows, columns)
                {
                    matrices.insert(
                        words[offset + 1],
                        MatrixType {
                            component,
                            scope,
                            rows,
                            columns,
                            use_kind: constants
                                .get(&words[offset + 6])
                                .copied()
                                .unwrap_or(words[offset + 6]),
                        },
                    );
                }
            }
            OP_COOPERATIVE_MATRIX_LOAD_KHR if count >= 3 => {
                loads.insert(words[offset + 1]); // Result type.
                value_types.insert(words[offset + 2], words[offset + 1]);
            }
            OP_COOPERATIVE_MATRIX_STORE_KHR if count >= 3 => {
                // The stored matrix value is operand 2 (pointer, object, ...).
                // Retaining its type ties the required store to the same
                // accumulator/result type that the admitted MulAdd produces.
                if let Some(matrix_type) = value_types.get(&words[offset + 2]).copied() {
                    stores.insert(matrix_type);
                }
            }
            OP_COOPERATIVE_MATRIX_MUL_ADD_KHR if count >= 6 => {
                // Result type, result, A, B, C.
                mul_adds.push((
                    words[offset + 1],
                    words[offset + 3],
                    words[offset + 4],
                    words[offset + 5],
                ));
            }
            _ => {}
        }
        offset = end;
    }

    let uses_route = capability
        || extension
        || saw_matrix_type
        || !loads.is_empty()
        || !stores.is_empty()
        || !mul_adds.is_empty();
    if uses_route && !declared {
        return Err(Error::Shader(
            "cooperative-matrix SPIR-V requires experimental-cooperative-matrix".into(),
        ));
    }
    if !declared {
        return Ok(());
    }
    if !capability
        || !extension
        || matrices.is_empty()
        || loads.is_empty()
        || stores.is_empty()
        || mul_adds.is_empty()
    {
        return Err(Error::Shader(
            "cooperative-matrix route requires KHR extension, capability, types, loads, multiply-add, and store".into(),
        ));
    }

    // A MulAdd's result type and its three operand types provide the complete
    // A/B/C/result tuple. Requiring a single exact native tuple prevents a
    // shader from being admitted merely because the device supports a related
    // component or dimension combination.
    let mut exact_tuple = false;
    for (result_type_id, a_value, b_value, c_value) in mul_adds {
        let Some(result) = matrices.get(&result_type_id).copied() else {
            continue;
        };
        let Some(a_type_id) = value_types.get(&a_value).copied() else {
            continue;
        };
        let Some(b_type_id) = value_types.get(&b_value).copied() else {
            continue;
        };
        let Some(c_type_id) = value_types.get(&c_value).copied() else {
            continue;
        };
        let Some(a) = matrices.get(&a_type_id).copied() else {
            continue;
        };
        let Some(b) = matrices.get(&b_type_id).copied() else {
            continue;
        };
        let Some(c) = matrices.get(&c_type_id).copied() else {
            continue;
        };
        if result.use_kind != USE_ACCUMULATOR
            || a.use_kind != USE_A
            || b.use_kind != USE_B
            || c.use_kind != USE_ACCUMULATOR
            || a.scope != result.scope
            || b.scope != result.scope
            || c.scope != result.scope
            || a.rows != c.rows
            || a.columns != b.rows
            || b.columns != c.columns
            || c.rows != result.rows
            || c.columns != result.columns
            || !loads.contains(&a_type_id)
            || !loads.contains(&b_type_id)
            || !stores.contains(&result_type_id)
        {
            continue;
        }
        exact_tuple |= device
            .0
            .cooperative_matrix_configurations
            .iter()
            .any(|configuration| {
                configuration.m == c.rows
                    && configuration.n == c.columns
                    && configuration.k == a.columns
                    && configuration.a_type == a.component
                    && configuration.b_type == b.component
                    && configuration.c_type == c.component
                    && configuration.result_type == result.component
                    && !configuration.saturating_accumulation
                    && configuration.scope == c.scope
            });
    }
    if !exact_tuple {
        return Err(Error::Unsupported(
            "Vulkan cooperative-matrix SPIR-V tuple is not supported by this device",
        ));
    }
    Ok(())
}

/// Restricts the first admitted cooperative-matrix route to the layout and
/// stride exercised by its real-device fixture. The tuple capability alone
/// does not establish that a device accepts every legal matrix memory layout.
///
/// `OpCooperativeMatrixLoadKHR` names its layout at operand 4 and optional
/// stride at operand 5; `OpCooperativeMatrixStoreKHR` names them at operands
/// 3 and 4. Both must resolve to RowMajorKHR and the explicit stride 16 used
/// by the selected 16x16x16 StorageBuffer artifact. Its pointers are exactly
/// two-index access chains into storage-buffer structs: member zero at
/// `Offset 0`, then runtime-array element zero. Keeping this finite
/// provenance check prevents an otherwise valid layout/stride pair from
/// silently admitting a nonzero or dynamic byte offset.
fn validate_cooperative_matrix_layout(words: &[u32]) -> Result<()> {
    const OP_CONSTANT: u32 = 43;
    const OP_TYPE_INT: u32 = 21;
    const OP_TYPE_POINTER: u32 = 32;
    const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
    const OP_TYPE_STRUCT: u32 = 30;
    const OP_VARIABLE: u32 = 59;
    const OP_DECORATE: u32 = 71;
    const OP_ACCESS_CHAIN: u32 = 65;
    const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;
    const OP_MEMBER_DECORATE: u32 = 72;
    const OP_COOPERATIVE_MATRIX_LOAD_KHR: u32 = 4457;
    const OP_COOPERATIVE_MATRIX_STORE_KHR: u32 = 4458;
    const DECORATION_OFFSET: u32 = 35;
    const DECORATION_BLOCK: u32 = 2;
    const COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR: u32 = 0;
    const SELECTED_STRIDE: u32 = 16;
    const STORAGE_CLASS_STORAGE_BUFFER: u32 = 12;

    if words.len() < 5 || words[0] != 0x0723_0203 {
        return Err(Error::Shader("invalid SPIR-V header".into()));
    }
    let mut integer_widths = std::collections::HashMap::<u32, u32>::new();
    let mut constants = std::collections::HashMap::<u32, u32>::new();
    let mut pointer_types = std::collections::HashMap::<u32, (u32, u32)>::new();
    let mut runtime_arrays = std::collections::HashMap::<u32, u32>::new();
    let mut structs = std::collections::HashMap::<u32, Vec<u32>>::new();
    let mut variables = std::collections::HashMap::<u32, u32>::new();
    let mut block_structs = std::collections::HashSet::<u32>::new();
    let mut access_chains = std::collections::HashMap::<u32, (u32, u32, Vec<u32>)>::new();
    let mut member_offsets = std::collections::HashMap::<(u32, u32), u32>::new();
    let mut layouts_strides_and_pointers = Vec::<(u32, Option<u32>, u32)>::new();
    let mut offset = 5usize;
    while offset < words.len() {
        let header = words[offset];
        let count = (header >> 16) as usize;
        let opcode = header & 0xffff;
        let end = offset
            .checked_add(count)
            .filter(|end| count != 0 && *end <= words.len())
            .ok_or_else(|| Error::Shader("invalid SPIR-V instruction length".into()))?;
        match opcode {
            OP_TYPE_INT if count == 4 => {
                integer_widths.insert(words[offset + 1], words[offset + 2]);
            }
            // The selected route uses 32-bit scalar ids for layout, stride,
            // and access-chain indexes. Do not truncate a wider literal's
            // high words when proving a zero byte offset.
            OP_CONSTANT if count == 4 && integer_widths.get(&words[offset + 1]) == Some(&32) => {
                constants.insert(words[offset + 2], words[offset + 3]);
            }
            OP_TYPE_POINTER if count >= 4 => {
                pointer_types.insert(words[offset + 1], (words[offset + 2], words[offset + 3]));
            }
            OP_TYPE_RUNTIME_ARRAY if count >= 3 => {
                runtime_arrays.insert(words[offset + 1], words[offset + 2]);
            }
            OP_TYPE_STRUCT if count >= 2 => {
                structs.insert(words[offset + 1], words[offset + 2..end].to_vec());
            }
            OP_VARIABLE if count >= 4 => {
                variables.insert(words[offset + 2], words[offset + 1]);
            }
            OP_DECORATE
                if count == 3 && words[offset + 2] == DECORATION_BLOCK =>
            {
                block_structs.insert(words[offset + 1]);
            }
            OP_ACCESS_CHAIN | OP_IN_BOUNDS_ACCESS_CHAIN if count >= 5 => {
                access_chains.insert(
                    words[offset + 2],
                    (
                        words[offset + 1],
                        words[offset + 3],
                        words[offset + 4..end].to_vec(),
                    ),
                );
            }
            OP_MEMBER_DECORATE if count >= 5 && words[offset + 3] == DECORATION_OFFSET => {
                member_offsets.insert((words[offset + 1], words[offset + 2]), words[offset + 4]);
            }
            // Result type, result, pointer, layout, optional stride, memory operands.
            OP_COOPERATIVE_MATRIX_LOAD_KHR => {
                if count < 5 {
                    return Err(Error::Shader(
                        "malformed OpCooperativeMatrixLoadKHR instruction".into(),
                    ));
                }
                let stride = if count >= 6 {
                    Some(words[offset + 5])
                } else {
                    None
                };
                layouts_strides_and_pointers.push((words[offset + 4], stride, words[offset + 3]));
            }
            // Pointer, object, layout, optional stride, memory operands.
            OP_COOPERATIVE_MATRIX_STORE_KHR => {
                if count < 4 {
                    return Err(Error::Shader(
                        "malformed OpCooperativeMatrixStoreKHR instruction".into(),
                    ));
                }
                let stride = if count >= 5 {
                    Some(words[offset + 4])
                } else {
                    None
                };
                layouts_strides_and_pointers.push((words[offset + 3], stride, words[offset + 1]));
            }
            _ => {}
        }
        offset = end;
    }
    for (layout_id, stride_id, pointer_id) in layouts_strides_and_pointers {
        let layout = constants.get(&layout_id).copied();
        let stride = stride_id.and_then(|id| constants.get(&id).copied());
        if layout != Some(COOPERATIVE_MATRIX_LAYOUT_ROW_MAJOR_KHR)
            || stride != Some(SELECTED_STRIDE)
        {
            return Err(Error::Unsupported(
                "Vulkan cooperative-matrix route requires RowMajorKHR with explicit stride 16",
            ));
        }
        let Some((result_type, base_id, indexes)) = access_chains.get(&pointer_id) else {
            return Err(Error::Unsupported(
                "Vulkan cooperative-matrix route requires a statically zero-offset access chain",
            ));
        };
        let Some(variable_pointer_type) = variables.get(base_id) else {
            return Err(Error::Unsupported(
                "Vulkan cooperative-matrix route requires a storage-buffer access-chain base",
            ));
        };
        let Some((storage_class, struct_id)) = pointer_types.get(variable_pointer_type) else {
            return Err(Error::Unsupported(
                "Vulkan cooperative-matrix route requires a typed storage-buffer base",
            ));
        };
        let Some(members) = structs.get(struct_id) else {
            return Err(Error::Unsupported(
                "Vulkan cooperative-matrix route requires a storage-buffer struct base",
            ));
        };
        let Some(runtime_array_id) = members.first() else {
            return Err(Error::Unsupported(
                "Vulkan cooperative-matrix route requires a runtime-array storage-buffer member",
            ));
        };
        let result_pointee = pointer_types
            .get(result_type)
            .filter(|(storage_class, _)| *storage_class == STORAGE_CLASS_STORAGE_BUFFER)
            .map(|(_, pointee)| *pointee);
        let pointer_matches_artifact = *storage_class == STORAGE_CLASS_STORAGE_BUFFER
            && members.len() == 1
            && block_structs.contains(struct_id)
            && runtime_arrays.get(runtime_array_id) == result_pointee.as_ref()
            && member_offsets.get(&(*struct_id, 0)) == Some(&0)
            && indexes.len() == 2
            && indexes
                .iter()
                .all(|index| constants.get(index).copied() == Some(0));
        if !pointer_matches_artifact {
            return Err(Error::Unsupported(
                "Vulkan cooperative-matrix route requires an explicit zero byte-offset access chain",
            ));
        }
    }
    Ok(())
}

fn validate_barycentric_route(
    device: &VulkanDevice,
    desc: &api::ShaderDesc,
    words: &[u32],
) -> Result<()> {
    let khr = spirv_has_extension(words, b"SPV_KHR_fragment_shader_barycentric");
    let nv = spirv_has_extension(words, b"SPV_NV_fragment_shader_barycentric");
    let amd = spirv_has_extension(words, b"SPV_AMD_shader_explicit_vertex_parameter");
    let standard_declared = desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_BARYCENTRICS);
    let amd_declared = desc
        .required_features
        .iter()
        .any(|feature| feature == api::SHADER_FEATURE_BARYCENTRICS_AMD);
    if ((khr || nv) && !standard_declared) || (amd && !amd_declared) {
        return Err(Error::Shader(
            "SPIR-V barycentric extension must be declared in required_features".into(),
        ));
    }
    if standard_declared
        && ((khr && !device.0.shader_barycentrics_khr)
            || (nv && !device.0.shader_barycentrics_nv)
            || (!khr && !nv))
    {
        return Err(Error::Unsupported(
            "Vulkan SPIR-V KHR/NV barycentric extension route",
        ));
    }
    if amd_declared && !(amd && device.0.shader_barycentrics_amd) {
        return Err(Error::Unsupported(
            "Vulkan SPIR-V AMD barycentric extension route",
        ));
    }
    Ok(())
}

fn spirv_has_extension(words: &[u32], extension: &[u8]) -> bool {
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return false;
    }
    let mut offset = 5usize;
    while offset < words.len() {
        let instruction = words[offset];
        let word_count = (instruction >> 16) as usize;
        let opcode = instruction & 0xffff;
        let Some(end) = offset.checked_add(word_count) else {
            return false;
        };
        if word_count == 0 || end > words.len() {
            return false;
        }
        if opcode == 10 && word_count >= 2 {
            let name = words[offset + 1..end]
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .take_while(|byte| *byte != 0)
                .collect::<Vec<_>>();
            if name == extension {
                return true;
            }
        }
        offset = end;
    }
    false
}

#[path = "shader_push.rs"]
mod push;
#[path = "shader_sampler.rs"]
mod sampler;
pub(crate) use sampler::usage as sampler_usage;
#[path = "shader_wgsl.rs"]
mod wgsl;
pub(crate) use wgsl::{compile_variant, create_wgsl};
#[path = "shader_numeric.rs"]
mod numeric;
#[path = "shader_subgroup.rs"]
mod subgroup;
