//! WGSL validation, Metal Shading Language output, and native library creation.
//!
//! Each pipeline assigns stage-local slots in group and binding order, with
//! separate buffer, texture and sampler namespaces. Push constants follow
//! the resource buffer slots. Vertex inputs occupy the final buffer slots.
//! Native Metal shaders must use the same layout-derived indices; WGSL is
//! translated per pipeline to apply those indices automatically.

#[path = "shader_variant.rs"]
pub(crate) mod variant;

use std::future::{Ready, ready};

use fluxel_rhi_core::api::{
    self, ShaderBackend, ShaderDesc, ShaderFormat, ShaderRoute, ShaderTarget,
};

use super::{Metal, MetalDevice, MetalShader};

/// Preserve the RHI's WGSL push-constant spelling when using Naga's renamed
/// immediate address space. Only a `var<...>` address-space token is changed.
pub(crate) fn normalize_push_address_space(source: &str) -> String {
    let mut result = String::with_capacity(source.len());
    let mut cursor = 0;
    while let Some(relative) = source[cursor..].find("push_constant") {
        let start = cursor + relative;
        let before = source[..start].trim_end();
        let prefix = before.strip_suffix('<').map(str::trim_end);
        let after = &source[start + "push_constant".len()..];
        if prefix.is_some_and(|p| p.ends_with("var")) && after.trim_start().starts_with('>') {
            result.push_str(&source[cursor..start]);
            result.push_str("immediate");
            cursor = start + "push_constant".len();
        } else {
            let end = start + "push_constant".len();
            result.push_str(&source[cursor..end]);
            cursor = end;
        }
    }
    result.push_str(&source[cursor..]);
    result
}

impl ShaderBackend for Metal {
    type DirectFuture<'a> = Ready<api::Result<MetalShader>>;
    type NagaFuture<'a> = Ready<api::Result<MetalShader>>;

    fn shader_route(_: &MetalDevice, desc: &ShaderDesc) -> ShaderRoute {
        route(desc)
    }

    fn compile_direct<'a>(
        device: &'a MetalDevice,
        desc: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::DirectFuture<'a> {
        ready(match desc.format {
            ShaderFormat::MetalLibrary => load_library(device, &desc.bytes, desc.entries.clone()),
            ShaderFormat::MetalSource => {
                compile_library(device, &desc.bytes, desc.entries.clone(), Vec::new(), false)
            }
            _ => Err(api::Error::Unsupported("Metal direct shader format")),
        })
    }

    fn compile_naga<'a>(
        device: &'a MetalDevice,
        desc: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::NagaFuture<'a> {
        let result = (|| {
            let source = std::str::from_utf8(&desc.bytes)
                .map_err(|error| api::Error::Shader(error.to_string()))?;
            let normalized = normalize_push_address_space(source);
            let module = naga::front::wgsl::parse_str(&normalized)
                .map_err(|error| api::Error::Shader(error.emit_to_string(&normalized)))?;
            if module.types.iter().any(|(_, ty)| {
                matches!(
                    ty.inner,
                    naga::TypeInner::AccelerationStructure { .. }
                        | naga::TypeInner::RayQuery { .. }
                )
            }) && !ray_query_supported(device)
            {
                return Err(api::Error::Unsupported("Metal device ray query"));
            }
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .map_err(|error| api::Error::Shader(error.to_string()))?;
            validate_entry_points(&module, desc)?;
            #[cfg(target_vendor = "apple")]
            {
                let _ = device;
                Ok(MetalShader {
                    _library: None,
                    entries: desc.entries.clone(),
                    entry_names: Vec::new(),
                    wgsl_source: Some(desc.bytes.clone()),
                })
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                let _ = device;
                Err(api::Error::Unsupported("Metal requires an Apple target"))
            }
        })();
        ready(result)
    }
}

fn route(desc: &ShaderDesc) -> ShaderRoute {
    let target = match desc.format {
        ShaderFormat::Wgsl => ShaderTarget {
            format: ShaderFormat::MetalSource,
            environment: "Metal Shading Language 2.0".into(),
            abi_key: "fluxel-metal-stage-slots-v2".into(),
        },
        ShaderFormat::MetalSource => ShaderTarget {
            format: ShaderFormat::MetalLibrary,
            environment: "Metal native library compiler".into(),
            abi_key: "fluxel-metal-stage-slots-v2".into(),
        },
        ShaderFormat::MetalLibrary => ShaderTarget {
            format: ShaderFormat::MetalLibrary,
            environment: "Metal precompiled library".into(),
            abi_key: "fluxel-metal-stage-slots-v2".into(),
        },
        _ => return ShaderRoute::Unsupported,
    };
    match desc.format {
        ShaderFormat::Wgsl => ShaderRoute::Naga(target),
        ShaderFormat::MetalSource | ShaderFormat::MetalLibrary => ShaderRoute::Direct(target),
        _ => unreachable!(),
    }
}

#[cfg(target_vendor = "apple")]
pub(super) fn ray_query_supported(device: &MetalDevice) -> bool {
    use objc2::runtime::NSObjectProtocol as _;
    use objc2_metal::MTLDevice as _;
    let native = &device._state._device;
    native.respondsToSelector(objc2::sel!(supportsRaytracing)) && native.supportsRaytracing()
}

#[cfg(not(target_vendor = "apple"))]
fn ray_query_supported(_: &MetalDevice) -> bool {
    false
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
impl MetalShader {
    pub(crate) fn native_entry_name<'a>(&'a self, name: &'a str) -> &'a str {
        self.entry_names
            .iter()
            .find(|(original, _)| original == name)
            .map_or(name, |(_, translated)| translated.as_str())
    }
}

/// A metallib is opaque; the caller supplies the native entry names and must
/// compile resource arguments for the advertised fixed-slot binding ABI.
/// Converter-produced argument-buffer layouts require a separate pipeline ABI.
#[cfg(target_vendor = "apple")]
fn load_library(
    device: &MetalDevice,
    bytes: &[u8],
    entries: Vec<api::ShaderEntryPoint>,
) -> api::Result<MetalShader> {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLDevice as _, MTLFunction as _, MTLFunctionType, MTLLibrary as _};

    if bytes.is_empty() || entries.is_empty() {
        return Err(api::Error::InvalidState);
    }
    let data = dispatch2::DispatchData::from_bytes(bytes);
    let library = device
        ._state
        ._device
        .newLibraryWithData_error(&data)
        .map_err(|error| api::Error::Shader(format!("Metal shader library: {error:?}")))?;
    for entry in &entries {
        let function = library
            .newFunctionWithName(&NSString::from_str(&entry.name))
            .ok_or_else(|| {
                api::Error::Shader(format!("Metal library entry {:?} not found", entry.name))
            })?;
        let expected = match entry.stage {
            api::ShaderStage::Vertex => Some(MTLFunctionType::Vertex),
            api::ShaderStage::Fragment => Some(MTLFunctionType::Fragment),
            api::ShaderStage::Compute => Some(MTLFunctionType::Kernel),
            api::ShaderStage::Task => Some(MTLFunctionType::Object),
            api::ShaderStage::Mesh => Some(MTLFunctionType::Mesh),
            // Converter-generated ray functions may have different native
            // types depending on compilation mode. Pipeline linking must
            // validate their types against the selected converter ABI.
            _ => None,
        };
        if expected.is_some_and(|ty| function.functionType() != ty) {
            return Err(api::Error::Shader(format!(
                "Metal library entry {:?} has a mismatched stage",
                entry.name
            )));
        }
    }
    Ok(MetalShader {
        _library: Some(library),
        entries,
        entry_names: Vec::new(),
        wgsl_source: None,
    })
}

#[cfg(not(target_vendor = "apple"))]
fn load_library(
    _: &MetalDevice,
    _: &[u8],
    _: Vec<api::ShaderEntryPoint>,
) -> api::Result<MetalShader> {
    Err(api::Error::Unsupported("Metal requires an Apple target"))
}

#[cfg(target_vendor = "apple")]
fn compile_library(
    device: &MetalDevice,
    bytes: &[u8],
    entries: Vec<api::ShaderEntryPoint>,
    entry_names: Vec<(String, String)>,
    ray_query: bool,
) -> api::Result<MetalShader> {
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCompileOptions, MTLDevice as _, MTLLanguageVersion};

    let source =
        std::str::from_utf8(bytes).map_err(|error| api::Error::Shader(error.to_string()))?;
    let source = NSString::from_str(source);
    let options = ray_query.then(|| {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_4);
        options
    });
    let library = device
        ._state
        ._device
        .newLibraryWithSource_options_error(&source, options.as_deref())
        .map_err(|error| api::Error::Shader(format!("Metal shader library: {error:?}")))?;
    Ok(MetalShader {
        _library: Some(library),
        entries,
        entry_names,
        wgsl_source: None,
    })
}

#[cfg(not(target_vendor = "apple"))]
fn compile_library(
    _: &MetalDevice,
    _: &[u8],
    _: Vec<api::ShaderEntryPoint>,
    _: Vec<(String, String)>,
    _: bool,
) -> api::Result<MetalShader> {
    Err(api::Error::Unsupported("Metal requires an Apple target"))
}

#[cfg(test)]
mod tests {
    use super::{route, validate_entry_points};
    use fluxel_rhi_core::api::{self, ShaderDesc, ShaderEntryPoint, ShaderFormat};
    use std::sync::Arc;

    #[test]
    fn metallib_has_a_direct_route_without_changing_source_routes() {
        let mut desc = ShaderDesc {
            label: None,
            format: ShaderFormat::MetalLibrary,
            bytes: Arc::from([1]),
            entries: vec![],
            required_features: vec![],
        };
        assert!(matches!(route(&desc), api::ShaderRoute::Direct(target)
            if target.format == ShaderFormat::MetalLibrary && target.abi_key == "fluxel-metal-stage-slots-v2"));
        desc.format = ShaderFormat::MetalSource;
        assert!(matches!(route(&desc), api::ShaderRoute::Direct(_)));
        desc.format = ShaderFormat::Wgsl;
        assert!(matches!(route(&desc), api::ShaderRoute::Naga(_)));
        desc.format = ShaderFormat::Dxil;
        assert!(matches!(route(&desc), api::ShaderRoute::Unsupported));
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
}
