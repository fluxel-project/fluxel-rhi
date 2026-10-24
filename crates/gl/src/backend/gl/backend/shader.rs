//! Static shader SPI routing, direct-source compilation, and common shader publication.
//!
//! Naga-specific WGSL parsing and GLSL generation live in `shader/wgsl.rs`;
//! this module chooses that route from the active profile and Core contract.
//! Compilation is a cold path and creates no program cache or command stream.

use std::future::{Ready, ready};

use fluxel_rhi_core::api::{
    self, Error, Result, ShaderBackend, ShaderDesc, ShaderRoute, ShaderTarget,
};

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
use super::spi::native_error;
use super::types::{GlNagaBindingName, GlShader, GlShaderInner};
use super::{Gl, GlDevice};

mod glsl_source;
use glsl_source::{apply_embedded_precision_defaults, apply_public_clip_depth_abi};

mod direct_constants;
pub(crate) mod spirv;

mod binding_array;
mod binding_array_dynamic;
mod binding_array_dynamic_buffer;
mod binding_array_dynamic_load;
mod binding_array_dynamic_sample;
mod binding_array_dynamic_sampled_load;
mod binding_array_dynamic_sampled_query;
mod binding_array_dynamic_storage_query;
mod binding_array_helper;
mod pairs;
mod remap;
mod runtime_binding_arrays;
mod specialize;
mod texture_num_levels;
mod texture_num_samples;
mod wgsl;

impl ShaderBackend for Gl {
    type DirectFuture<'a> = Ready<Result<GlShader>>;
    type NagaFuture<'a> = Ready<Result<GlShader>>;

    fn shader_route(device: &GlDevice, desc: &ShaderDesc) -> ShaderRoute {
        let profile = shader_profile(device);
        let compute = desc
            .entries
            .iter()
            .any(|entry| entry.stage == api::ShaderStage::Compute);
        // The static SPI type is present for every GL build, but this exact
        // adopted context may not have a proved compute route.  Refuse
        // the shader before choosing a compiler target so GL 4.0--4.2 and
        // WebGL2 never appear to accept an executable compute pipeline.
        if compute
            && !matches!(desc.format, api::ShaderFormat::Wgsl)
            && !wgsl::compute_supported(device)
        {
            return ShaderRoute::Unsupported;
        }
        if matches!(desc.format, api::ShaderFormat::Wgsl) {
            return profile.map_or(ShaderRoute::Unsupported, |profile| {
                ShaderRoute::Naga(ShaderTarget {
                    format: api::ShaderFormat::Wgsl,
                    environment: naga_environment(profile, compute).into(),
                    abi_key: "fluxel-gl-position-yz-row-order-v2".into(),
                })
            });
        }

        let Some(profile) = profile else {
            return ShaderRoute::Unsupported;
        };
        let [entry] = desc.entries.as_slice() else {
            return ShaderRoute::Unsupported;
        };
        if crate::backend::gl::translate::lower_shader_source(profile, desc, entry).is_err() {
            return ShaderRoute::Unsupported;
        }
        let format = match desc.format {
            api::ShaderFormat::Glsl(version) => api::ShaderFormat::Glsl(version),
            api::ShaderFormat::GlslEs(version) => api::ShaderFormat::GlslEs(version),
            api::ShaderFormat::SpirV
                if matches!(
                    profile,
                    crate::backend::gl::profile::GlProfile::Desktop { .. }
                ) && spirv_supported(device) =>
            {
                api::ShaderFormat::SpirV
            }
            _ => return ShaderRoute::Unsupported,
        };
        ShaderRoute::Direct(ShaderTarget {
            format,
            environment: if matches!(desc.format, api::ShaderFormat::SpirV) {
                "opengl-spirv-1.0".into()
            } else {
                naga_environment(profile, false).into()
            },
            // Direct SPIR-V cannot be source-rewritten like direct GLSL. Its
            // producer must target the native GL coordinate/resource ABI; do
            // not advertise the GLSL/WGSL adaptation ABI for that artifact.
            abi_key: if matches!(desc.format, api::ShaderFormat::SpirV) {
                "fluxel-gl-native-spirv-v1"
            } else {
                "fluxel-gl-position-yz-row-order-v2"
            }
            .into(),
        })
    }

    fn compile_direct<'a>(
        device: &'a GlDevice,
        desc: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::DirectFuture<'a> {
        ready(compile_direct(device, desc))
    }

    fn compile_naga<'a>(
        device: &'a GlDevice,
        desc: &'a ShaderDesc,
        _: &'a ShaderTarget,
    ) -> Self::NagaFuture<'a> {
        ready(wgsl::compile_wgsl(device, desc))
    }
}

fn spirv_supported(device: &GlDevice) -> bool {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        return device
            .0
            .with_native("query GL SPIR-V shader capability", |provider| {
                Ok(provider.supports_spirv())
            })
            .unwrap_or(false);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = device;
        false
    }
}

/// Resolves the profile once without placing a platform context object in Core.
fn shader_profile(device: &GlDevice) -> Option<crate::backend::gl::profile::GlProfile> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;

        return device
            .0
            .with_native(
                "select GL shader profile",
                |provider| Ok(provider.profile()),
            )
            .ok();
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        return Some(device.0.owner.profile());
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = device;
        None
    }
}

fn naga_environment(
    profile: crate::backend::gl::profile::GlProfile,
    compute: bool,
) -> &'static str {
    match (profile, compute) {
        (crate::backend::gl::profile::GlProfile::Desktop { .. }, true) => "glsl-430",
        (crate::backend::gl::profile::GlProfile::Desktop { .. }, false) => "glsl-400",
        (crate::backend::gl::profile::GlProfile::Embedded { .. }, true) => "gles-310",
        (crate::backend::gl::profile::GlProfile::Embedded { .. }, false) => "gles-300",
        (crate::backend::gl::profile::GlProfile::WebGl2, false) => "webgl2-essl-300",
        (crate::backend::gl::profile::GlProfile::WebGl2, true) => "webgl2-essl-300",
    }
}

fn compile_direct(device: &GlDevice, desc: &ShaderDesc) -> Result<GlShader> {
    let [entry] = desc.entries.as_slice() else {
        return Err(Error::Unsupported("one direct GL shader entry point"));
    };
    let profile = shader_profile(device).ok_or(Error::Unsupported("GL platform context"))?;
    let (_, mut source) = crate::backend::gl::translate::lower_shader_source(profile, desc, entry)?;
    if source.spirv.is_none() {
        apply_embedded_precision_defaults(&mut source);
        apply_public_clip_depth_abi(&mut source)?;
    }
    refresh_source_key(&mut source);
    // Direct GLSL constants are supplied by `PipelineStage`, not `ShaderDesc`.
    // Compiling here would make an otherwise valid `#define FOO` use fail
    // before materialization has injected FOO.  Program creation below always
    // compiles the retained immutable source after stage specialization.
    Ok(GlShader::from_inner(GlShaderInner {
        state: device.0.clone(),
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        id: None,
        source: Some(source),
        wgsl: None,
        naga_bindings: Vec::new(),
        push_uniforms: Vec::new(),
        mip_uniforms: Vec::new(),
    }))
}

fn publish_shader(
    device: &GlDevice,
    source: crate::backend::gl::api::GlShaderSource,
    naga_bindings: Vec<GlNagaBindingName>,
    push_uniforms: Vec<super::push::PushUniform>,
    mip_uniforms: Vec<super::types::GlMipUniform>,
    operation: &'static str,
) -> Result<GlShader> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlShaderApi as _;

        let id = device
            .0
            .with_native(operation, |provider| provider.create_shader(&source))
            .map_err(native_error)?;
        return Ok(GlShader::from_inner(GlShaderInner {
            state: device.0.clone(),
            id: Some(id),
            source: Some(source),
            wgsl: None,
            naga_bindings,
            push_uniforms,
            mip_uniforms,
        }));
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        // Browser program creation compiles both stage sources together so it
        // can return link diagnostics. Retaining the lowered source here is
        // therefore the native shader object's truthful equivalent.
        let _ = operation;
        return Ok(GlShader::from_inner(GlShaderInner {
            state: device.0.clone(),
            source: Some(source),
            wgsl: None,
            naga_bindings,
            push_uniforms,
            mip_uniforms,
        }));
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (device, source, naga_bindings, push_uniforms, operation);
        Err(Error::Unsupported("GL platform context"))
    }
}

pub(super) fn materialize_stage(
    device: &GlDevice,
    shader: &GlShader,
    entry: &str,
    stage: api::ShaderStage,
    constants: &[(String, api::SpecializationValue)],
    multiview: Option<std::num::NonZeroU32>,
    layout: &super::types::GlPipelineLayout,
) -> Result<GlShader> {
    if !std::rc::Rc::ptr_eq(&device.0, &shader.state) {
        return Err(Error::InvalidState);
    }
    if let Some(desc) = &shader.wgsl {
        let selected = desc
            .entries
            .iter()
            .find(|item| item.name == entry && item.stage == stage)
            .ok_or_else(|| {
                Error::Shader(format!("WGSL entry point {entry} not declared for stage"))
            })?;
        if stage == api::ShaderStage::Compute && !wgsl::compute_supported(device) {
            return Err(Error::Unsupported("GL compute capability"));
        }
        let mut selected_desc = desc.clone();
        selected_desc.entries = vec![selected.clone()];
        let profile = shader_profile(device).ok_or(Error::Unsupported("GL platform context"))?;
        let parameters = if stage == api::ShaderStage::Vertex {
            draw_parameters(device)?
        } else {
            DrawParameters::None
        };
        let (image_limit, storage_buffer_limit) = shader_resource_limits(device, stage)?;
        let sampled_limit = shader_sampled_limit(device, stage)?;
        let uniform_limit = shader_uniform_limit(device, stage)?;
        let (source, bindings, push_uniforms, mip_uniforms) = wgsl::lower_wgsl_with_limits(
            profile,
            &selected_desc,
            constants,
            parameters,
            multiview,
            image_limit,
            storage_buffer_limit,
            sampled_limit,
            uniform_limit,
            Some(layout),
        )?;
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        if matches!(
            profile,
            crate::backend::gl::profile::GlProfile::Embedded { major: 3, minor: 1 }
        ) && bindings.iter().any(|binding| {
            matches!(
                binding.resource_type,
                Some(super::types::GlNagaResourceType::SampledTexture {
                    dimension: api::TextureViewDimension::D2Array,
                    multisampled: true,
                    ..
                })
            )
        }) {
            use crate::backend::gl::api::GlFamilyApi as _;
            let supported = device
                .0
                .with_native("query multisample array shader capability", |provider| {
                    Ok(provider
                        .discovery()
                        .extensions()
                        .raw_reported_names()
                        .any(|name| name == "GL_OES_texture_storage_multisample_2d_array"))
                })
                .map_err(native_error)?;
            if !supported {
                return Err(Error::Unsupported(
                    "GLES multisample array shader extension",
                ));
            }
        }
        return publish_shader(
            device,
            source,
            bindings,
            push_uniforms,
            mip_uniforms,
            "compile WGSL-derived GL shader",
        );
    }
    let source = shader.source.as_ref().ok_or(Error::InvalidState)?;
    if source.entry_point != entry || source.stage != wgsl::gl_stage(stage)? {
        return Err(Error::InvalidState);
    }
    if !constants.is_empty() || multiview.is_some() {
        let mut source = source.clone();
        if source.spirv.is_some() {
            spirv::specialize(&mut source, constants)?;
            refresh_source_key(&mut source);
            if multiview.is_some() {
                return Err(Error::Unsupported("GL SPIR-V multiview ABI"));
            }
        } else {
            direct_constants::apply(&mut source, constants)?;
        }
        if multiview.is_some() {
            source.text = wgsl::lower_multiview(source.text, stage, multiview)?;
            source.source_hash = crate::backend::gl::api::GlShaderKey(source_key(
                source.text.as_bytes(),
                &source.entry_point,
            ));
        }
        return publish_shader(
            device,
            source,
            shader.naga_bindings.clone(),
            shader.push_uniforms.clone(),
            shader.mip_uniforms.clone(),
            "compile multiview GLSL stage",
        );
    }
    Ok(shader.clone())
}

fn refresh_source_key(source: &mut crate::backend::gl::api::GlShaderSource) {
    if let Some(spirv) = &source.spirv {
        let mut bytes = Vec::with_capacity(spirv.words.len() * 4 + spirv.specialization.len() * 8);
        for word in &spirv.words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        for (id, value) in &spirv.specialization {
            bytes.extend_from_slice(&id.to_le_bytes());
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        source.source_hash =
            crate::backend::gl::api::GlShaderKey(source_key(&bytes, &source.entry_point));
    } else {
        source.source_hash = crate::backend::gl::api::GlShaderKey(source_key(
            source.text.as_bytes(),
            &source.entry_point,
        ));
    }
}

fn shader_resource_limits(device: &GlDevice, stage: api::ShaderStage) -> Result<(u32, u32)> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        return device
            .0
            .with_native("query GL shader image limits", |provider| {
                let limits = provider.discovery().limits();
                let per_stage = match stage {
                    api::ShaderStage::Vertex => limits.max_vertex_image_uniforms,
                    api::ShaderStage::Fragment => limits.max_fragment_image_uniforms,
                    api::ShaderStage::Compute => limits.max_compute_image_uniforms,
                    _ => 0,
                };
                let storage_stage = match stage {
                    api::ShaderStage::Vertex => limits.max_vertex_storage_blocks,
                    api::ShaderStage::Fragment => limits.max_fragment_storage_blocks,
                    api::ShaderStage::Compute => limits.max_compute_storage_blocks,
                    _ => 0,
                };
                Ok((
                    limits
                        .max_image_units
                        .min(limits.max_combined_image_units)
                        .min(per_stage),
                    limits
                        .max_storage_buffer_bindings
                        .min(limits.max_combined_storage_blocks)
                        .min(storage_stage),
                ))
            })
            .map_err(native_error);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = (device, stage);
        Ok((0, 0))
    }
}

fn shader_uniform_limit(device: &GlDevice, stage: api::ShaderStage) -> Result<u32> {
    let choose = |limits: &crate::backend::gl::api::GlLimits| {
        let per_stage = match stage {
            api::ShaderStage::Vertex => limits.max_vertex_uniform_blocks,
            api::ShaderStage::Fragment => limits.max_fragment_uniform_blocks,
            api::ShaderStage::Compute => limits.max_compute_uniform_blocks,
            _ => 0,
        };
        per_stage
            .min(limits.max_combined_uniform_blocks)
            .min(limits.max_uniform_buffer_bindings)
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        return device
            .0
            .with_native("query GL shader uniform limits", |provider| {
                Ok(choose(&provider.discovery().limits()))
            })
            .map_err(native_error);
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        return Ok(choose(&device.0.owner.discovery_snapshot().limits()));
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (device, choose);
        Ok(0)
    }
}

fn shader_sampled_limit(device: &GlDevice, stage: api::ShaderStage) -> Result<u32> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        return device
            .0
            .with_native("query GL shader texture limits", |provider| {
                let snapshot = provider.discovery();
                let limits = snapshot.limits();
                let per_stage = match stage {
                    api::ShaderStage::Vertex => limits.max_vertex_texture_image_units,
                    api::ShaderStage::Fragment => limits.max_fragment_texture_image_units,
                    api::ShaderStage::Compute => limits.max_compute_texture_image_units,
                    _ => 0,
                };
                Ok(per_stage.min(limits.max_combined_texture_image_units))
            })
            .map_err(native_error);
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let discovery = device.0.owner.discovery_snapshot();
        let limits = discovery.limits();
        let per_stage = match stage {
            api::ShaderStage::Vertex => limits.max_vertex_texture_image_units,
            api::ShaderStage::Fragment => limits.max_fragment_texture_image_units,
            api::ShaderStage::Compute => limits.max_compute_texture_image_units,
            _ => 0,
        };
        return Ok(per_stage.min(limits.max_combined_texture_image_units));
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (device, stage);
        Ok(0)
    }
}

/// Vertex builtin support is a context capability, not a profile assumption.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DrawParameters {
    None,
    Core,
    Arb,
}

pub(super) fn draw_parameters(device: &GlDevice) -> Result<DrawParameters> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        return device
            .0
            .with_native("query GL shader draw parameters", |provider| {
                let profile = provider.profile();
                if let crate::backend::gl::api::GlFamilyProfile::Desktop { major, minor } = profile
                {
                    if major > 4 || major == 4 && minor >= 6 {
                        return Ok(DrawParameters::Core);
                    }
                    if provider
                        .discovery()
                        .extensions()
                        .raw_reported_names()
                        .any(|name| name == "GL_ARB_shader_draw_parameters")
                    {
                        return Ok(DrawParameters::Arb);
                    }
                }
                Ok(DrawParameters::None)
            })
            .map_err(native_error);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = device;
        Ok(DrawParameters::None)
    }
}

pub(super) fn source_key(bytes: &[u8], entry: &str) -> [u8; 32] {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes.iter().chain(entry.as_bytes()) {
        state = (state ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3);
    }
    let mut key = [0; 32];
    for index in 0..4 {
        key[index * 8..(index + 1) * 8]
            .copy_from_slice(&state.wrapping_add(index as u64).to_le_bytes());
    }
    key
}

#[cfg(test)]
mod tests;
