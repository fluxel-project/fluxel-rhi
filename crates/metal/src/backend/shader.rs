//! WGSL validation, Metal Shading Language output, and native library creation.
//!
//! Each pipeline assigns stage-local slots in group and binding order, with
//! separate buffer, texture and sampler namespaces. Push constants follow
//! the resource buffer slots. Vertex inputs occupy the final buffer slots.
//! Native Metal shaders must use the same layout-derived indices; WGSL is
//! translated per pipeline to apply those indices automatically.

#[path = "shader_variant/mod.rs"]
pub(crate) mod variant;

use fluxel_rhi_core::api::{self, ShaderDesc, ShaderFormat, ShaderRoute, ShaderTarget};
use fluxel_rhi_core::backend::ShaderBackend;

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
    type DirectFuture<'a> = super::cold::ColdFuture<MetalShader>;
    type NagaFuture<'a> = super::cold_cpu::CpuFuture<MetalShader>;

    fn shader_route(_: &MetalDevice, desc: &ShaderDesc) -> ShaderRoute {
        route(desc)
    }

    fn compile_direct<'a>(
        device: &'a MetalDevice,
        desc: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::DirectFuture<'a> {
        match desc.format {
            // A precompiled metallib has no native compilation work to defer.
            ShaderFormat::MetalLibrary => super::cold::ColdFuture::ready(load_library(
                device,
                &desc.bytes,
                desc.entries.clone(),
            )),
            ShaderFormat::MetalSource => compile_library_async(
                device,
                desc.bytes.clone(),
                desc.entries.clone(),
                Vec::new(),
                false,
            ),
            _ => super::cold::ColdFuture::ready(Err(api::Error::Unsupported(
                "Metal direct shader format",
            ))),
        }
    }

    fn compile_naga<'a>(
        device: &'a MetalDevice,
        desc: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::NagaFuture<'a> {
        // Naga is CPU-only but can be expensive. Never run it on a caller or
        // Metal callback queue; the bounded worker owns only Send Rust data.
        let ray_query = ray_query_supported(device);
        let bytes = desc.bytes.clone();
        let entries = desc.entries.clone();
        super::cold_cpu::spawn_cpu(move || compile_naga_cpu(bytes, entries, ray_query))
    }
}

fn compile_naga_cpu(
    bytes: std::sync::Arc<[u8]>,
    entries: Vec<api::ShaderEntryPoint>,
    ray_query_supported: bool,
) -> api::Result<MetalShader> {
    let source =
        std::str::from_utf8(&bytes).map_err(|error| api::Error::Shader(error.to_string()))?;
    let normalized = normalize_push_address_space(source);
    let module = naga::front::wgsl::parse_str(&normalized)
        .map_err(|error| api::Error::Shader(error.emit_to_string(&normalized)))?;
    if module.types.iter().any(|(_, ty)| {
        matches!(
            ty.inner,
            naga::TypeInner::AccelerationStructure { .. } | naga::TypeInner::RayQuery { .. }
        )
    }) && !ray_query_supported
    {
        return Err(api::Error::Unsupported("Metal device ray query"));
    }
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| api::Error::Shader(error.to_string()))?;
    validate_entry_points(&module, &entries)?;
    #[cfg(target_vendor = "apple")]
    {
        Ok(MetalShader {
            _library: None,
            entries,
            entry_names: Vec::new(),
            wgsl_source: Some(bytes),
        })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = (bytes, entries);
        Err(api::Error::Unsupported("Metal requires an Apple target"))
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

fn validate_entry_points(
    module: &naga::Module,
    entries: &[api::ShaderEntryPoint],
) -> api::Result<()> {
    if entries.is_empty() {
        return Err(api::Error::InvalidState);
    }
    for requested in entries {
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
pub(crate) fn compile_library_async(
    device: &MetalDevice,
    bytes: std::sync::Arc<[u8]>,
    entries: Vec<api::ShaderEntryPoint>,
    entry_names: Vec<(String, String)>,
    ray_query: bool,
) -> super::cold::ColdFuture<MetalShader> {
    use objc2::rc::Retained;
    use objc2_foundation::NSString;
    use objc2_metal::{MTLCompileOptions, MTLDevice as _, MTLLanguageVersion};

    let source = match std::str::from_utf8(&bytes) {
        Ok(source) => NSString::from_str(source),
        Err(error) => {
            return super::cold::ColdFuture::ready(Err(api::Error::Shader(error.to_string())));
        }
    };
    let options = ray_query.then(|| {
        let options = MTLCompileOptions::new();
        options.setLanguageVersion(MTLLanguageVersion::Version2_4);
        options
    });
    let (future, completer) = super::cold::pending();
    let metadata = std::sync::Mutex::new(Some((entries, entry_names)));
    let callback_source = source.clone();
    let callback_options = options.clone();
    // Keep source/options and the result metadata alive until Metal invokes the
    // handler. A dropped Rust future only stops observation; Metal may still
    // compile and safely release these captures on its callback queue.
    let callback = block2::RcBlock::new(move |library, error| {
        let _source = &callback_source;
        let _options = &callback_options;
        let result = if let Some(library) = unsafe { Retained::retain(library) } {
            match metadata
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .take()
            {
                Some((entries, entry_names)) => Ok(MetalShader {
                    _library: Some(library),
                    entries,
                    entry_names,
                    wgsl_source: None,
                }),
                None => Err(api::Error::Backend(
                    "Metal shader completion invoked twice".into(),
                )),
            }
        } else {
            let error = unsafe { Retained::retain(error) };
            Err(api::Error::Shader(format!(
                "Metal shader library: {error:?}"
            )))
        };
        completer.finish(result);
    });
    // SAFETY: `callback` has the exact generated completion-handler ABI and
    // Metal copies it before this method returns. Its captures retain every
    // Objective-C input and all Rust result metadata through callback entry.
    unsafe {
        device
            ._state
            ._device
            .newLibraryWithSource_options_completionHandler(
                &source,
                options.as_deref(),
                block2::RcBlock::as_ptr(&callback),
            );
    }
    future
}

#[cfg(not(target_vendor = "apple"))]
pub(crate) fn compile_library_async(
    _: &MetalDevice,
    _: std::sync::Arc<[u8]>,
    _: Vec<api::ShaderEntryPoint>,
    _: Vec<(String, String)>,
    _: bool,
) -> super::cold::ColdFuture<MetalShader> {
    super::cold::ColdFuture::ready(Err(api::Error::Unsupported(
        "Metal requires an Apple target",
    )))
}

#[cfg(test)]
#[path = "../tests/backend/shader.rs"]
mod tests;
