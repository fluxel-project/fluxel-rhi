//! WGSL validation, Metal Shading Language output, and native library creation.
//!
//! The fixed binding ABI assigns seven Metal slots to each RHI bind group and
//! supports four groups. This keeps group binding independent of unrelated
//! shader globals; out-of-range slots fail during shader creation. Pipeline
//! construction remains a separate SPI operation and is not implied here.

use std::future::{Ready, ready};

use fluxel_rhi_core::api::{
    self, ShaderBackend, ShaderDesc, ShaderFormat, ShaderRoute, ShaderTarget,
};

use super::{Metal, MetalDevice, MetalShader};

const GROUP_STRIDE: u32 = 7;
const MAX_GROUPS: u32 = 4;
const MAX_BINDINGS_PER_GROUP: u32 = 7;
const MAX_SLOT: u32 = GROUP_STRIDE * MAX_GROUPS - 1;

impl ShaderBackend for Metal {
    type DirectFuture<'a> = Ready<api::Result<MetalShader>>;
    type NagaFuture<'a> = Ready<api::Result<MetalShader>>;

    fn shader_route(_: &MetalDevice, desc: &ShaderDesc) -> ShaderRoute {
        let target = match desc.format {
            ShaderFormat::Wgsl => ShaderTarget {
                format: ShaderFormat::MetalSource,
                environment: "Metal Shading Language 2.0".into(),
                abi_key: "fluxel-metal-bind-group-slots-v1".into(),
            },
            ShaderFormat::MetalSource => ShaderTarget {
                format: ShaderFormat::MetalLibrary,
                environment: "Metal native library compiler".into(),
                abi_key: "fluxel-metal-source-v1".into(),
            },
            _ => return ShaderRoute::Unsupported,
        };
        match desc.format {
            ShaderFormat::Wgsl => ShaderRoute::Naga(target),
            ShaderFormat::MetalSource => ShaderRoute::Direct(target),
            _ => unreachable!(),
        }
    }

    fn compile_direct<'a>(
        device: &'a MetalDevice,
        desc: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::DirectFuture<'a> {
        ready(compile_library(device, &desc.bytes, desc.entries.clone()))
    }

    fn compile_naga<'a>(
        device: &'a MetalDevice,
        desc: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::NagaFuture<'a> {
        ready(
            translate_wgsl(&desc.bytes, desc).and_then(|source| {
                compile_library(device, source.as_bytes(), desc.entries.clone())
            }),
        )
    }
}

fn translate_wgsl(bytes: &[u8], desc: &ShaderDesc) -> api::Result<String> {
    let source =
        std::str::from_utf8(bytes).map_err(|error| api::Error::Shader(error.to_string()))?;
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|error| api::Error::Shader(error.emit_to_string(source)))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| api::Error::Shader(error.to_string()))?;
    validate_entry_points(&module, desc)?;

    let mut bindings = naga::back::msl::BindingMap::new();
    for (_, global) in module.global_variables.iter() {
        let Some(binding) = global.binding.as_ref() else {
            continue;
        };
        let slot = metal_slot(binding.group, binding.binding)?;
        // Naga chooses the native Metal argument namespace from the first
        // populated target field, so only the resource's actual namespace may
        // be present. Sharing the slot number does not mean sharing argument
        // namespaces on MTL*CommandEncoder.
        let (buffer, texture, sampler) = match module.types[global.ty].inner {
            naga::TypeInner::Image { .. } => (None, Some(slot), None),
            naga::TypeInner::Sampler { .. } => (
                None,
                None,
                Some(naga::back::msl::BindSamplerTarget::Resource(slot)),
            ),
            _ => (Some(slot), None, None),
        };
        bindings.insert(
            naga::ResourceBinding {
                group: binding.group,
                binding: binding.binding,
            },
            naga::back::msl::BindTarget {
                buffer,
                texture,
                sampler,
                external_texture: None,
                mutable: false,
            },
        );
    }

    let mut options = naga::back::msl::Options {
        fake_missing_bindings: false,
        ..Default::default()
    };
    for entry in &desc.entries {
        options.per_entry_point_map.insert(
            entry.name.clone(),
            naga::back::msl::EntryPointResources {
                resources: bindings.clone(),
                ..Default::default()
            },
        );
    }
    let (source, _) = naga::back::msl::write_string(
        &module,
        &info,
        &options,
        &naga::back::msl::PipelineOptions::default(),
    )
    .map_err(|error| api::Error::Shader(error.to_string()))?;
    Ok(source)
}

fn metal_slot(group: u32, binding: u32) -> api::Result<u8> {
    if group >= MAX_GROUPS || binding >= MAX_BINDINGS_PER_GROUP {
        return Err(api::Error::Unsupported(
            "Metal bind group or binding index exceeds ABI",
        ));
    }
    let slot = group
        .checked_mul(GROUP_STRIDE)
        .and_then(|base| base.checked_add(binding))
        .filter(|slot| *slot <= MAX_SLOT)
        .ok_or(api::Error::Unsupported("Metal binding index exceeds ABI"))?;
    Ok(slot as u8)
}

fn validate_entry_points(module: &naga::Module, desc: &ShaderDesc) -> api::Result<()> {
    if desc.entries.is_empty() {
        return Err(api::Error::InvalidState);
    }
    for requested in &desc.entries {
        let stage = match requested.stage {
            api::ShaderStage::Vertex => naga::ShaderStage::Vertex,
            api::ShaderStage::Fragment => naga::ShaderStage::Fragment,
            api::ShaderStage::Compute => naga::ShaderStage::Compute,
            _ => return Err(api::Error::Unsupported("Metal shader stage")),
        };
        if !module
            .entry_points
            .iter()
            .any(|entry| entry.name == requested.name && entry.stage == stage)
        {
            return Err(api::Error::Shader(format!(
                "WGSL entry point {:?} does not match the RHI descriptor",
                requested.name
            )));
        }
    }
    Ok(())
}

#[cfg(target_vendor = "apple")]
fn compile_library(
    device: &MetalDevice,
    bytes: &[u8],
    entries: Vec<api::ShaderEntryPoint>,
) -> api::Result<MetalShader> {
    use objc2_foundation::NSString;
    use objc2_metal::MTLDevice as _;

    let source =
        std::str::from_utf8(bytes).map_err(|error| api::Error::Shader(error.to_string()))?;
    let source = NSString::from_str(source);
    let library = device
        ._state
        ._device
        .newLibraryWithSource_options_error(&source, None)
        .map_err(|error| api::Error::Shader(format!("Metal shader library: {error:?}")))?;
    Ok(MetalShader {
        _library: library,
        entries,
    })
}

#[cfg(not(target_vendor = "apple"))]
fn compile_library(
    _: &MetalDevice,
    _: &[u8],
    _: Vec<api::ShaderEntryPoint>,
) -> api::Result<MetalShader> {
    Err(api::Error::Unsupported("Metal requires an Apple target"))
}

#[cfg(test)]
mod tests {
    use super::{metal_slot, translate_wgsl, validate_entry_points};
    use fluxel_rhi_core::api::{self, ShaderDesc, ShaderEntryPoint, ShaderFormat};
    use std::sync::Arc;

    #[test]
    fn group_slot_mapping_stays_inside_metal_abi() {
        assert_eq!(metal_slot(0, 0).unwrap(), 0);
        assert_eq!(metal_slot(1, 0).unwrap(), 7);
        assert_eq!(metal_slot(3, 6).unwrap(), 27);
        assert!(matches!(metal_slot(0, 7), Err(api::Error::Unsupported(_))));
        assert!(matches!(metal_slot(4, 0), Err(api::Error::Unsupported(_))));
    }

    #[test]
    fn wgsl_entry_point_stage_must_match_core_descriptor() {
        let source = "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }";
        let module = naga::front::wgsl::parse_str(source).unwrap();
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_bytes()),
            entries: vec![ShaderEntryPoint {
                name: "vs".into(),
                stage: api::ShaderStage::Vertex,
                workgroup_size: None,
            }],
            required_features: Vec::new(),
        };
        assert!(validate_entry_points(&module, &desc).is_ok());
        let mut mismatch = desc;
        mismatch.entries[0].stage = api::ShaderStage::Fragment;
        assert!(matches!(
            validate_entry_points(&module, &mismatch),
            Err(api::Error::Shader(_))
        ));
    }

    #[test]
    fn wgsl_translation_uses_the_declared_metal_buffer_slot() {
        let source = "struct Uniforms { color: vec4<f32>, }\n@group(1) @binding(0) var<uniform> uniforms: Uniforms;\n@vertex fn vs() -> @builtin(position) vec4<f32> { return uniforms.color; }";
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_bytes()),
            entries: vec![ShaderEntryPoint {
                name: "vs".into(),
                stage: api::ShaderStage::Vertex,
                workgroup_size: None,
            }],
            required_features: Vec::new(),
        };
        let msl = translate_wgsl(source.as_bytes(), &desc).unwrap();
        assert!(msl.contains("[[buffer(7)]]"), "{msl}");
    }

    #[test]
    fn wgsl_translation_lowers_compute_storage_texture_to_metal_texture_slot() {
        let source = "@group(0) @binding(1) var output_tex: texture_storage_2d<rgba8unorm, write>;\n@compute @workgroup_size(8, 8, 1) fn cs(@builtin(global_invocation_id) id: vec3<u32>) { textureStore(output_tex, vec2<i32>(id.xy), vec4<f32>(1.0)); }";
        let desc = ShaderDesc {
            label: None,
            format: ShaderFormat::Wgsl,
            bytes: Arc::from(source.as_bytes()),
            entries: vec![ShaderEntryPoint {
                name: "cs".into(),
                stage: api::ShaderStage::Compute,
                workgroup_size: Some([8, 8, 1]),
            }],
            required_features: Vec::new(),
        };
        let msl = translate_wgsl(source.as_bytes(), &desc).unwrap();
        assert!(msl.contains("[[texture(1)]]"), "{msl}");
    }
}
