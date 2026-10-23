//! Raster-pipeline creation for the immediate GL backend.
//!
//! Linking and VAO allocation happen on the cold creation path. The resulting
//! pipeline retains only native object identities and immutable fixed state;
//! draw calls are still issued directly by the encoder.

use std::future::{Ready, ready};

use fluxel_rhi_core::api::{
    self, Error, PipelineBackend, RasterPipelineDesc, Result, VertexFormat, VertexStepMode,
};

use super::spi::native_error;
use super::types::{GlPipelineLayout, GlRasterPipeline, GlRasterPipelineInner};
use super::{Gl, GlDevice};

impl PipelineBackend for Gl {
    type RasterPipelineFuture<'a> = Ready<Result<GlRasterPipeline>>;

    fn create_raster_pipeline<'a>(
        device: &'a GlDevice,
        desc: &'a RasterPipelineDesc<Self>,
    ) -> Self::RasterPipelineFuture<'a> {
        ready(create_raster_pipeline(device, desc))
    }
}

fn create_raster_pipeline(
    device: &GlDevice,
    desc: &RasterPipelineDesc<Gl>,
) -> Result<GlRasterPipeline> {
    let fragment = desc.fragment.as_ref();
    let uses_src1 = desc
        .color_targets
        .iter()
        .flatten()
        .filter_map(|target| target.blend)
        .any(|blend| {
            [
                blend.color.src_factor,
                blend.color.dst_factor,
                blend.alpha.src_factor,
                blend.alpha.dst_factor,
            ]
            .iter()
            .any(|f| matches!(f, api::BlendFactor::Src1 | api::BlendFactor::OneMinusSrc1))
        });
    let dual_output = if let Some(fragment) = fragment {
        // SAFETY: Core supplies a device-validated pipeline shader.
        if let Some(wgsl) = &unsafe { fragment.module.native_ref() }.wgsl {
            let source = std::str::from_utf8(&wgsl.bytes)
                .map_err(|_| Error::Shader("WGSL is not UTF-8".into()))?;
            let module = naga::front::wgsl::parse_str(source)
                .map_err(|e| Error::Shader(e.emit_to_string(source)))?;
            let entry = module
                .entry_points
                .iter()
                .find(|ep| ep.name == fragment.entry && ep.stage == naga::ShaderStage::Fragment)
                .ok_or_else(|| Error::Shader("fragment entry not found".into()))?;
            let mut found = false;
            if let Some(result) = &entry.function.result {
                let check = |binding: &Option<naga::Binding>| {
                    matches!(
                        binding,
                        Some(naga::Binding::Location {
                            blend_src: Some(1),
                            ..
                        })
                    )
                };
                found |= check(&result.binding);
                if let naga::TypeInner::Struct { members, .. } = &module.types[result.ty].inner {
                    found |= members.iter().any(|member| check(&member.binding));
                }
            }
            found
        } else {
            false
        }
    } else {
        false
    };
    let direct_fragment = fragment.is_some_and(|stage| {
        // SAFETY: Core retains and validates this shader during pipeline creation.
        unsafe { stage.module.native_ref() }.wgsl.is_none()
    });
    if uses_src1 || dual_output {
        if !dual_output && !direct_fragment {
            return Err(Error::Unsupported(
                "dual-source blending requires a secondary output",
            ));
        }
        #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
        let limit = device
            .0
            .with_native("query dual-source blending", |provider| {
                Ok(provider.dual_source_limit())
            })
            .map_err(native_error)?;
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        let limit = device.0.owner.dual_source_limit()?;
        #[cfg(not(any(
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl",
            all(target_arch = "wasm32", feature = "webgl2")
        )))]
        let limit = 0u32;
        if limit == 0 {
            return Err(Error::Unsupported("GL dual-source blending"));
        }
        if desc.color_targets.len() > limit as usize
            || desc.color_targets.len() != 1
            || desc.color_targets[0].is_none()
        {
            return Err(Error::Unsupported(
                "GL dual-source blending requires a single color target",
            ));
        }
    }
    let (topology, cull_mode, front_face) =
        crate::backend::gl::translate_raster::lower_primitive(desc.primitive)?;
    let depth_stencil = desc.depth_stencil.map(lower_depth_stencil).transpose()?;
    let rasterization = crate::backend::gl::api::GlRasterizationState {
        polygon_mode: match desc.primitive.polygon_mode {
            api::PolygonMode::Fill => crate::backend::gl::api::GlPolygonMode::Fill,
            api::PolygonMode::Line => crate::backend::gl::api::GlPolygonMode::Line,
            api::PolygonMode::Point => crate::backend::gl::api::GlPolygonMode::Point,
        },
        depth_clamp: desc.primitive.unclipped_depth,
        conservative: desc.primitive.conservative,
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    if !device
        .0
        .with_native("validate GL rasterization modes", |provider| {
            Ok(provider.supports_rasterization(rasterization))
        })
        .map_err(native_error)?
    {
        return Err(Error::Unsupported("native GL/GLES rasterization modes"));
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        use crate::backend::gl::api::{GlKnownExtension, GlPolygonMode};
        let discovery = device.0.owner.discovery_snapshot();
        let extensions = discovery.extensions();
        if rasterization.conservative
            || rasterization.polygon_mode == GlPolygonMode::Point
            || (rasterization.polygon_mode == GlPolygonMode::Line
                && !extensions.is_acquired(GlKnownExtension::WebglPolygonMode))
            || (rasterization.depth_clamp
                && !extensions.is_acquired(GlKnownExtension::ExtDepthClamp))
        {
            return Err(Error::Unsupported("WebGL2 rasterization modes"));
        }
    }
    if let Some(views) = desc.multiview {
        let limit = super::instance::multiview_limit(&device.0)?;
        if limit == 0 {
            return Err(Error::Unsupported("GL OVR multiview"));
        }
        if views == 0 || views > limit {
            return Err(Error::InvalidRange);
        }
    }
    let color_targets = desc
        .color_targets
        .iter()
        .map(|target| match target {
            Some(target) => Ok(crate::backend::gl::api::GlColorTargetState {
                write_mask: target.write_mask,
                blend: target
                    .blend
                    .map(|blend| {
                        Ok(crate::backend::gl::api::GlBlendState {
                            color: crate::backend::gl::translate_raster::lower_blend_component(
                                blend.color,
                                false,
                            )?,
                            alpha: crate::backend::gl::translate_raster::lower_blend_component(
                                blend.alpha,
                                true,
                            )?,
                        })
                    })
                    .transpose()?,
            }),
            None => Ok(crate::backend::gl::api::GlColorTargetState {
                write_mask: 0,
                blend: None,
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    let vertex_layout = lower_vertex_layout(&desc.vertex_buffers)?;
    let multisample = crate::backend::gl::translate_raster::lower_multisample(desc.multisample)?;
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        let (mask_supported, clamp_supported) = device
            .0
            .with_native("validate GL coverage and bias", |provider| {
                Ok((
                    provider.supports_sample_mask(),
                    provider.supports_polygon_offset_clamp(),
                ))
            })
            .map_err(native_error)?;
        if multisample.sample_mask != u64::MAX && !mask_supported {
            return Err(Error::Unsupported("native GL/GLES sample mask"));
        }
        if depth_stencil.is_some_and(|state| f32::from_bits(state.depth_bias[2]) != 0.0)
            && !clamp_supported
        {
            return Err(Error::Unsupported("native GL/GLES polygon offset clamp"));
        }
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        validate_webgl2_fixed_state(&color_targets, depth_stencil, multisample)?;
        if depth_stencil.is_some_and(|state| f32::from_bits(state.depth_bias[2]) != 0.0)
            && !device
                .0
                .owner
                .discovery_snapshot()
                .extensions()
                .is_acquired(crate::backend::gl::api::GlKnownExtension::ExtPolygonOffsetClamp)
        {
            return Err(Error::Unsupported("WebGL2 polygon offset clamp"));
        }
    }
    // SAFETY: Core verified shader/layout provenance before calling this SPI.
    // We only read immutable backend source/layout descriptors while those
    // public handles are borrowed for this synchronous creation call.
    let vertex = super::shader::materialize_stage(
        device,
        unsafe { desc.vertex.module.native_ref() },
        &desc.vertex.entry,
        api::ShaderStage::Vertex,
        &desc.vertex.constants,
        desc.multiview.and_then(std::num::NonZeroU32::new),
    )?;
    let fragment = fragment
        .map(|stage| {
            super::shader::materialize_stage(
                device,
                unsafe { stage.module.native_ref() },
                &stage.entry,
                api::ShaderStage::Fragment,
                &stage.constants,
                desc.multiview.and_then(std::num::NonZeroU32::new),
            )
        })
        .transpose()?;
    let vertex = &vertex;
    let fragment = fragment.as_ref();
    // SAFETY: Core retains the pipeline-layout handle for synchronous
    // pipeline creation. Copy its immutable ABI before native linking so the
    // returned pipeline can reject incompatible bind groups later.
    let layout = unsafe { desc.layout.native_ref() }.clone();
    let mut shaders = vec![vertex];
    if let Some(fragment) = fragment {
        shaders.push(fragment);
    }
    validate_naga_resource_types(&layout, &shaders)?;
    validate_combined_image_limits(device, &shaders)?;
    validate_combined_sampled_limits(device, &shaders)?;
    validate_combined_buffer_limits(device, &layout, &shaders)?;
    let push_uniforms = shaders
        .iter()
        .flat_map(|shader| shader.push_uniforms.iter().cloned())
        .collect::<Vec<_>>();
    let mip_uniforms = shaders
        .iter()
        .flat_map(|shader| shader.mip_uniforms.iter().cloned())
        .collect::<Vec<_>>();
    super::push::validate_uniforms(&push_uniforms)?;
    let mut native_layout = with_naga_binding_names(layout.native.clone(), &shaders)?;
    let mut vertex_source = vertex.source.clone().ok_or(Error::InvalidState)?;
    let mut fragment_source = fragment
        .and_then(|shader| shader.source.clone())
        .unwrap_or_else(|| {
            let mut source = vertex.source.clone().expect("materialized vertex source");
            source.stage = crate::backend::gl::api::GlShaderStage::Fragment;
            source.entry_point = "main".into();
            source.source_hash = crate::backend::gl::api::GlShaderKey([0x69; 32]);
            source.text = match source.dialect {
                crate::backend::gl::api::GlShaderDialect::Desktop { version } => {
                    format!("#version {version} core\nvoid main() {{}}\n")
                }
                crate::backend::gl::api::GlShaderDialect::Embedded { version } => {
                    format!("#version {version} es\nprecision highp float;\nvoid main() {{}}\n")
                }
            };
            source
        });
    harmonize_embedded_versions(&mut vertex_source, &mut fragment_source)?;
    remap_raster_image_units(
        &mut native_layout,
        vertex,
        &mut vertex_source,
        fragment,
        &mut fragment_source,
    )?;
    let descriptor = crate::backend::gl::api::GlProgramDescriptor {
        kind: crate::backend::gl::api::GlProgramKind::Raster {
            vertex: vertex_source,
            fragment: fragment_source,
        },
        layout: native_layout,
        debug_name: desc.label.clone(),
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::{GlShaderApi as _, GlVertexApi as _};
        let result = device
            .0
            .with_native("create GL raster pipeline", |provider| {
                // SAFETY: Core validated the cache's device identity and keeps
                // its handle borrowed throughout synchronous pipeline creation.
                let cache = desc.cache.as_ref().map(|cache| unsafe { cache.native_ref() });
                let (program, reflection) = super::cache::create_program(provider, &descriptor, cache)?;
                if uses_src1 && direct_fragment {
                    match provider.has_secondary_output(program) {
                        Ok(true) => {},
                        Ok(false) => { let _ = provider.destroy_program(program); return Err(crate::backend::gl::api::GlError::Unsupported { operation: "dual-source blending", reason: "linked fragment shader has no secondary output at location zero" }); },
                        Err(error) => { let _ = provider.destroy_program(program); return Err(error); }
                    }
                }
                match provider.create_vertex_array(&vertex_layout) {
                    Ok(vertex_array) => Ok((program, vertex_array, reflection)),
                    Err(error) => {
                        let _ = provider.destroy_program(program);
                        Err(error)
                    }
                }
            });
        let pipeline = result
            .map(|(program, vertex_array, reflection)| {
                GlRasterPipeline::from_inner(GlRasterPipelineInner {
                    mip_uniforms: mip_uniforms.clone(),
                    multiview: desc.multiview,
                    push_uniforms: push_uniforms.clone(),
                    state: device.0.clone(),
                    layout,
                    program,
                    vertex_array,
                    reflection,
                    vertex_layout: vertex_layout.clone(),
                    raster: crate::backend::gl::api::GlRasterState {
                        rasterization,
                        topology,
                        cull_mode,
                        front_face,
                        depth_stencil,
                        color_targets,
                        multisample,
                        // The active pass supplies its validated drawable size
                        // at `set_pipeline`; creation cannot guess it.
                        viewport: crate::backend::gl::api::GlViewport {
                            x: 0,
                            y: 0,
                            width: 0,
                            height: 0,
                            min_depth: 0.0f32.to_bits(),
                            max_depth: 1.0f32.to_bits(),
                        },
                        scissor: None,
                        blend_constant: [0; 4],
                    },
                })
            })
            .map_err(native_error)?;
        super::push::validate_active(
            &pipeline.push_uniforms,
            &pipeline.layout.push_ranges,
            |name| {
                device
                    .0
                    .with_native("validate GL push constants", |provider| {
                        provider.has_uniform(pipeline.program, name)
                    })
                    .map_err(native_error)
            },
        )?;
        return Ok(pipeline);
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            // WebGL owns program and VAO allocation on the browser thread.
            // They are created here, on the cold path, and are subsequently
            // selected by immediate raster verbs without a command cache.
            let (program, reflection) = device.0.owner.create_raster_program(&descriptor)?;
            super::push::validate_active(&push_uniforms, &layout.push_ranges, |name| {
                program.has_uniform(name)
            })?;
            let vertex_array = device.0.owner.create_vertex_array()?;
            return Ok(GlRasterPipeline::from_inner(GlRasterPipelineInner {
                mip_uniforms: mip_uniforms.clone(),
                multiview: desc.multiview,
                push_uniforms: push_uniforms.clone(),
                state: device.0.clone(),
                layout,
                browser_program: program,
                browser_vertex_array: vertex_array,
                reflection,
                vertex_layout: vertex_layout.clone(),
                raster: crate::backend::gl::api::GlRasterState {
                    rasterization,
                    topology,
                    cull_mode,
                    front_face,
                    depth_stencil,
                    color_targets,
                    multisample,
                    viewport: crate::backend::gl::api::GlViewport {
                        x: 0,
                        y: 0,
                        width: 0,
                        height: 0,
                        min_depth: 0.0f32.to_bits(),
                        max_depth: 1.0f32.to_bits(),
                    },
                    scissor: None,
                    blend_constant: [0; 4],
                },
            }));
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        {
            let _ = (
                device,
                descriptor,
                vertex_layout,
                topology,
                cull_mode,
                front_face,
                color_targets,
                multisample,
            );
            Err(Error::Unsupported("GL platform context"))
        }
    }
}

type UniformBinding = (u32, u32, Option<u32>);

fn check_uniform_limits(stages: &[Vec<UniformBinding>], points: u32, combined: u32) -> Result<()> {
    let mut unique = std::collections::HashSet::new();
    let mut stage_total = 0usize;
    for stage in stages {
        let active: std::collections::HashSet<_> = stage.iter().copied().collect();
        stage_total += active.len();
        unique.extend(active);
    }
    if unique.len() > points as usize || stage_total > combined as usize {
        return Err(Error::Unsupported(
            "GL combined shader buffer binding limit",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod uniform_limit_tests {
    use super::check_uniform_limits;

    #[test]
    fn counts_unique_points_and_each_stage_separately() {
        let shared = (0, 1, None);
        let array_first = (1, 0, Some(0));
        let array_second = (1, 0, Some(1));
        let stages = vec![
            vec![shared, shared, array_first],
            vec![shared, array_second],
        ];
        assert!(check_uniform_limits(&stages, 3, 4).is_ok());
        assert!(check_uniform_limits(&stages, 2, 4).is_err());
        assert!(check_uniform_limits(&stages, 3, 3).is_err());
    }
}

fn validate_combined_buffer_limits(
    device: &GlDevice,
    layout: &super::types::GlPipelineLayout,
    shaders: &[&super::types::GlShader],
) -> Result<()> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    let (points, combined, storage_points, storage_combined) = {
        use crate::backend::gl::api::GlFamilyApi as _;
        device
            .0
            .with_native("query combined GL uniform limits", |provider| {
                let limits = provider.discovery().limits();
                Ok((
                    limits.max_uniform_buffer_bindings,
                    limits.max_combined_uniform_blocks,
                    limits.max_storage_buffer_bindings,
                    limits.max_combined_storage_blocks,
                ))
            })
            .map_err(native_error)?
    };
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    let (points, combined, storage_points, storage_combined) = {
        let limits = device.0.owner.discovery_snapshot().limits();
        (
            limits.max_uniform_buffer_bindings,
            limits.max_combined_uniform_blocks,
            limits.max_storage_buffer_bindings,
            limits.max_combined_storage_blocks,
        )
    };
    #[cfg(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    ))]
    {
        for (storage, points, combined) in [
            (false, points, combined),
            (true, storage_points, storage_combined),
        ] {
            let stages: Vec<Vec<UniformBinding>> = shaders
                .iter()
                .map(|shader| {
                    shader
                        .naga_bindings
                        .iter()
                        .filter_map(|binding| {
                            let group = layout.groups.get(binding.group as usize)?;
                            let entry = group
                                .entries
                                .iter()
                                .find(|entry| entry.binding == binding.binding)?;
                            let matches_kind = if storage {
                                matches!(
                                    entry.slot,
                                    super::types::GlBindingSlot::StorageBuffer { .. }
                                )
                            } else {
                                matches!(
                                    entry.slot,
                                    super::types::GlBindingSlot::UniformBuffer { .. }
                                )
                            };
                            matches_kind.then_some((
                                binding.group,
                                binding.binding,
                                binding.element,
                            ))
                        })
                        .collect()
                })
                .collect();
            check_uniform_limits(&stages, points, combined)?;
        }
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (device, layout, shaders);
    }
    Ok(())
}

fn validate_combined_sampled_limits(
    device: &GlDevice,
    shaders: &[&super::types::GlShader],
) -> Result<()> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    let max_combined = {
        use crate::backend::gl::api::GlFamilyApi as _;
        device
            .0
            .with_native("query combined GL texture limit", |provider| {
                Ok(provider
                    .discovery()
                    .limits()
                    .max_combined_texture_image_units)
            })
            .map_err(native_error)?
    };
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    let max_combined = device
        .0
        .owner
        .discovery_snapshot()
        .limits()
        .max_combined_texture_image_units;
    #[cfg(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    ))]
    {
        let count: usize = shaders
            .iter()
            .map(|shader| {
                shader
                    .naga_bindings
                    .iter()
                    .filter(|binding| {
                        matches!(
                            binding.resource_type,
                            Some(super::types::GlNagaResourceType::SampledTexture { .. })
                        )
                    })
                    .map(|binding| &binding.name)
                    .collect::<std::collections::HashSet<_>>()
                    .len()
            })
            .sum();
        if count > max_combined as usize {
            return Err(Error::Unsupported("GL combined shader texture limit"));
        }
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (device, shaders);
    }
    Ok(())
}

fn validate_combined_image_limits(
    device: &GlDevice,
    shaders: &[&super::types::GlShader],
) -> Result<()> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlFamilyApi as _;
        let (max_units, max_combined) = device
            .0
            .with_native("query combined GL image limits", |provider| {
                let limits = provider.discovery().limits();
                Ok((limits.max_image_units, limits.max_combined_image_units))
            })
            .map_err(native_error)?;
        let mut unique = std::collections::HashSet::new();
        let mut combined = 0usize;
        for shader in shaders {
            let images: std::collections::HashSet<_> = shader
                .naga_bindings
                .iter()
                .filter_map(|binding| {
                    matches!(
                        binding.resource_type,
                        Some(super::types::GlNagaResourceType::StorageTexture { .. })
                    )
                    .then_some((
                        binding.group,
                        binding.binding,
                        binding.element,
                    ))
                })
                .collect();
            combined += images.len();
            unique.extend(images);
        }
        if unique.len() > max_units as usize || combined > max_combined as usize {
            return Err(Error::Unsupported("GL combined shader image limit"));
        }
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        let _ = (device, shaders);
    }
    Ok(())
}

fn harmonize_embedded_versions(
    vertex: &mut crate::backend::gl::api::GlShaderSource,
    fragment: &mut crate::backend::gl::api::GlShaderSource,
) -> Result<()> {
    use crate::backend::gl::api::GlShaderDialect;
    let (GlShaderDialect::Embedded { version: a }, GlShaderDialect::Embedded { version: b }) =
        (vertex.dialect, fragment.dialect)
    else {
        return Ok(());
    };
    let version = a.max(b);
    for source in [vertex, fragment] {
        if source.dialect == (GlShaderDialect::Embedded { version }) {
            continue;
        }
        let end = source.text.find('\n').ok_or(Error::InvalidState)?;
        if !source.text[..end].starts_with("#version ") {
            return Err(Error::InvalidState);
        }
        source
            .text
            .replace_range(..end, &format!("#version {version} es"));
        source.dialect = GlShaderDialect::Embedded { version };
        source.source_hash = crate::backend::gl::api::GlShaderKey(super::shader::source_key(
            source.text.as_bytes(),
            &source.entry_point,
        ));
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
fn validate_webgl2_fixed_state(
    colors: &[crate::backend::gl::api::GlColorTargetState],
    depth: Option<crate::backend::gl::api::GlDepthStencilState>,
    multisample: crate::backend::gl::api::GlMultisampleState,
) -> Result<()> {
    let _ = (colors, depth);
    if multisample.sample_mask != u64::MAX {
        return Err(Error::Unsupported("WebGL2 sample mask"));
    }
    Ok(())
}

/// Replaces placeholder ABI names with the names Naga actually emitted for
/// this pair of entry points. Native program reflection needs those names;
/// Naga's explicit storage-image unit is carried beside them for bind groups.
pub(crate) fn with_naga_binding_names(
    mut layout: crate::backend::gl::api::GlPipelineLayout,
    shaders: &[&super::types::GlShader],
) -> Result<crate::backend::gl::api::GlPipelineLayout> {
    let mut expanded = Vec::new();
    for binding in std::mem::take(&mut layout.bindings) {
        let mut elements: Vec<_> = shaders
            .iter()
            .flat_map(|shader| &shader.naga_bindings)
            .filter(|value| {
                value.group == binding.location.group && value.binding == binding.location.binding
            })
            .filter_map(|value| value.element)
            .collect();
        elements.sort_unstable();
        elements.dedup();
        if elements.is_empty() {
            expanded.push(binding);
        } else {
            for element in elements {
                if element >= binding.array_count {
                    return Err(Error::InvalidRange);
                }
                let mut item = binding.clone();
                item.element = Some(element);
                item.array_count = 1;
                expanded.push(item);
            }
        }
    }
    layout.bindings = expanded;
    for binding in &mut layout.bindings {
        let mut names = Vec::new();
        let mut image_units = Vec::new();
        for shader in shaders {
            image_units.extend(
                shader
                    .naga_bindings
                    .iter()
                    .filter(|value| {
                        value.group == binding.location.group
                            && value.binding == binding.location.binding
                            && value.element == binding.element
                    })
                    .filter_map(|value| value.image_unit),
            );
            names.extend(
                shader
                    .naga_bindings
                    .iter()
                    .filter(|value| {
                        value.group == binding.location.group
                            && value.binding == binding.location.binding
                            && value.element == binding.element
                    })
                    .map(|value| value.name.clone()),
            );
        }
        if !names.is_empty() {
            names.sort();
            names.dedup();
            binding.names = names;
        }
        if !image_units.is_empty() {
            image_units.sort_unstable();
            image_units.dedup();
            binding.image_unit = image_units.first().copied();
        }
    }
    Ok(layout)
}

pub(crate) fn validate_naga_resource_types(
    layout: &super::types::GlPipelineLayout,
    shaders: &[&super::types::GlShader],
) -> Result<()> {
    use super::types::{GlBindingSlot as L, GlNagaResourceType as R};
    use fluxel_rhi_core::api::TextureSampleType as S;
    for resource in shaders.iter().flat_map(|shader| &shader.naga_bindings) {
        let Some(resource_type) = resource.resource_type else {
            continue;
        };
        let entry = layout
            .groups
            .get(resource.group as usize)
            .and_then(|group| {
                group
                    .entries
                    .iter()
                    .find(|entry| entry.binding == resource.binding)
            })
            .ok_or(Error::InvalidState)?;
        let valid = match (resource_type, entry.slot) {
            (R::Sampler { comparison: a }, L::Sampler { comparison: b, .. }) => a == b,
            (
                R::StorageTexture {
                    dimension: a,
                    format: af,
                    access: aa,
                },
                L::StorageTexture {
                    dimension: b,
                    format: bf,
                    access: ba,
                },
            ) => a == b && af == bf && aa == ba,
            (
                R::SampledTexture {
                    dimension: a,
                    sample: sa,
                    multisampled: ma,
                },
                L::SampledTexture {
                    dimension: b,
                    sample: sb,
                    multisampled: mb,
                },
            ) => a == b && ma == mb && (sa == sb || (sa == S::Float && sb == S::UnfilterableFloat)),
            _ => false,
        };
        if !valid {
            return Err(Error::InvalidState);
        }
    }
    Ok(())
}

/// Image qualifiers are emitted before the other shader stage is known. Assign
/// one dense unit per logical binding at link time and rewrite only Naga's
/// corresponding image declarations in each source.
fn remap_raster_image_units(
    layout: &mut crate::backend::gl::api::GlPipelineLayout,
    vertex: &super::types::GlShader,
    vertex_source: &mut crate::backend::gl::api::GlShaderSource,
    fragment: Option<&super::types::GlShader>,
    fragment_source: &mut crate::backend::gl::api::GlShaderSource,
) -> Result<()> {
    let mut locations: Vec<_> = layout
        .bindings
        .iter()
        .filter(|binding| binding.image_unit.is_some())
        .map(|binding| {
            (
                binding.location.group,
                binding.location.binding,
                binding.element,
            )
        })
        .collect();
    locations.sort_unstable();
    locations.dedup();
    for binding in &mut layout.bindings {
        if binding.image_unit.is_some() {
            let location = (
                binding.location.group,
                binding.location.binding,
                binding.element,
            );
            binding.image_unit = Some(
                u32::try_from(
                    locations
                        .binary_search(&location)
                        .map_err(|_| Error::InvalidState)?,
                )
                .map_err(|_| Error::InvalidRange)?,
            );
        }
    }
    for (shader, source) in [(Some(vertex), vertex_source), (fragment, fragment_source)] {
        let Some(shader) = shader else { continue };
        for binding in shader
            .naga_bindings
            .iter()
            .filter(|binding| binding.image_unit.is_some())
        {
            let location = (binding.group, binding.binding, binding.element);
            let unit = locations
                .binary_search(&location)
                .map_err(|_| Error::InvalidState)?;
            let suffix = format!(" {};", binding.name);
            let old = format!("binding = {}", binding.image_unit.unwrap());
            let new = format!("binding = {unit}");
            let mut found = false;
            let mut text = String::with_capacity(source.text.len());
            for line in source.text.split_inclusive('\n') {
                if line.trim_end().ends_with(&suffix) && line.contains("uniform ") {
                    if found || !line.contains(&old) {
                        return Err(Error::InvalidState);
                    }
                    text.push_str(&line.replacen(&old, &new, 1));
                    found = true;
                } else {
                    text.push_str(line);
                }
            }
            if !found {
                return Err(Error::InvalidState);
            }
            source.text = text;
        }
        source.source_hash = crate::backend::gl::api::GlShaderKey(super::shader::source_key(
            source.text.as_bytes(),
            &source.entry_point,
        ));
    }
    Ok(())
}

fn lower_vertex_layout(
    values: &[api::VertexBufferLayout],
) -> Result<crate::backend::gl::api::GlVertexLayout> {
    let buffers = values
        .iter()
        .enumerate()
        .map(|(slot, value)| {
            Ok(crate::backend::gl::api::GlVertexBufferLayout {
                slot: u32::try_from(slot).map_err(|_| Error::InvalidRange)?,
                stride: u32::try_from(value.stride).map_err(|_| Error::InvalidRange)?,
                step_mode: match value.step_mode {
                    VertexStepMode::Vertex => crate::backend::gl::api::GlVertexStepMode::Vertex,
                    VertexStepMode::Instance => crate::backend::gl::api::GlVertexStepMode::Instance,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let attributes = values
        .iter()
        .enumerate()
        .flat_map(|(slot, buffer)| {
            buffer
                .attributes
                .iter()
                .map(move |attribute| (slot, attribute))
        })
        .map(|(slot, attribute)| {
            Ok(crate::backend::gl::api::GlVertexAttribute {
                location: attribute.location,
                buffer_slot: u32::try_from(slot).map_err(|_| Error::InvalidRange)?,
                format: lower_vertex_format(attribute.format)?,
                offset: u32::try_from(attribute.offset).map_err(|_| Error::InvalidRange)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(crate::backend::gl::api::GlVertexLayout {
        buffers,
        attributes,
    })
}

/// Lowers depth state without inventing a format conversion. Attachment
/// validation later checks that this declared format matches the pass view.
fn lower_depth_stencil(
    value: api::DepthStencilState,
) -> Result<crate::backend::gl::api::GlDepthStencilState> {
    if !matches!(
        value.format,
        api::TextureFormat::Depth16Unorm
            | api::TextureFormat::Depth24Plus
            | api::TextureFormat::Depth24PlusStencil8
            | api::TextureFormat::Depth32Float
            | api::TextureFormat::Depth32FloatStencil8
            | api::TextureFormat::Stencil8
    ) || !value.depth_bias_slope_scale.is_finite()
        || !value.depth_bias_clamp.is_finite()
    {
        return Err(Error::Unsupported("GL depth-stencil state"));
    }
    let face = |face: api::StencilFaceState| crate::backend::gl::api::GlStencilFaceState {
        compare: crate::backend::gl::translate_raster::lower_compare(face.compare),
        fail_op: lower_stencil_op(face.fail),
        depth_fail_op: lower_stencil_op(face.depth_fail),
        pass_op: lower_stencil_op(face.pass),
        read_mask: value.stencil_read_mask,
        write_mask: value.stencil_write_mask,
    };
    Ok(crate::backend::gl::api::GlDepthStencilState {
        depth_write_enabled: value.depth_write_enabled,
        depth_compare: crate::backend::gl::translate_raster::lower_compare(value.depth_compare),
        // Shared native/browser ABI: polygon offset factor, then units, then
        // clamp. Units are a float in GL even though Core exposes an integer.
        depth_bias: [
            value.depth_bias_slope_scale.to_bits(),
            (value.depth_bias_constant as f32).to_bits(),
            value.depth_bias_clamp.to_bits(),
        ],
        stencil_front: face(value.stencil_front),
        stencil_back: face(value.stencil_back),
        stencil_reference: 0,
    })
}

fn lower_stencil_op(value: api::StencilOperation) -> crate::backend::gl::api::GlStencilOperation {
    use crate::backend::gl::api::GlStencilOperation as Gl;
    match value {
        api::StencilOperation::Keep => Gl::Keep,
        api::StencilOperation::Zero => Gl::Zero,
        api::StencilOperation::Replace => Gl::Replace,
        api::StencilOperation::IncrementClamp => Gl::IncrementClamp,
        api::StencilOperation::DecrementClamp => Gl::DecrementClamp,
        api::StencilOperation::Invert => Gl::Invert,
        api::StencilOperation::IncrementWrap => Gl::IncrementWrap,
        api::StencilOperation::DecrementWrap => Gl::DecrementWrap,
    }
}

fn lower_vertex_format(value: VertexFormat) -> Result<crate::backend::gl::api::GlVertexFormat> {
    use crate::backend::gl::api::GlVertexFormat as Gl;
    Ok(match value {
        VertexFormat::Float32(1) => Gl::Float32,
        VertexFormat::Float32(2) => Gl::Float32x2,
        VertexFormat::Float32(3) => Gl::Float32x3,
        VertexFormat::Float32(4) => Gl::Float32x4,
        VertexFormat::Uint32(1) => Gl::Uint32,
        VertexFormat::Uint32(2) => Gl::Uint32x2,
        VertexFormat::Uint32(3) => Gl::Uint32x3,
        VertexFormat::Uint32(4) => Gl::Uint32x4,
        VertexFormat::Sint32(1) => Gl::Sint32,
        VertexFormat::Sint32(2) => Gl::Sint32x2,
        VertexFormat::Sint32(3) => Gl::Sint32x3,
        VertexFormat::Sint32(4) => Gl::Sint32x4,
        VertexFormat::Unorm8(2) => Gl::Unorm8x2,
        VertexFormat::Unorm8(4) => Gl::Unorm8x4,
        VertexFormat::Uint8(2) => Gl::Uint8x2,
        VertexFormat::Uint8(4) => Gl::Uint8x4,
        VertexFormat::Sint8(2) => Gl::Sint8x2,
        VertexFormat::Sint8(4) => Gl::Sint8x4,
        VertexFormat::Snorm8(2) => Gl::Snorm8x2,
        VertexFormat::Snorm8(4) => Gl::Snorm8x4,
        VertexFormat::Uint16(2) => Gl::Uint16x2,
        VertexFormat::Uint16(4) => Gl::Uint16x4,
        VertexFormat::Sint16(2) => Gl::Sint16x2,
        VertexFormat::Sint16(4) => Gl::Sint16x4,
        VertexFormat::Unorm16(2) => Gl::Unorm16x2,
        VertexFormat::Unorm16(4) => Gl::Unorm16x4,
        VertexFormat::Snorm16(2) => Gl::Snorm16x2,
        VertexFormat::Snorm16(4) => Gl::Snorm16x4,
        VertexFormat::Float16(2) => Gl::Float16x2,
        VertexFormat::Float16(4) => Gl::Float16x4,
        _ => return Err(Error::Unsupported("GL vertex format")),
    })
}

trait PipelineLayoutNativeRef {
    fn native_ref_safe(&self) -> &GlPipelineLayout;
}
impl PipelineLayoutNativeRef for api::PipelineLayout<Gl> {
    fn native_ref_safe(&self) -> &GlPipelineLayout {
        // SAFETY: this private backend method is called only while Core owns
        // the pipeline-layout handle for a pipeline being created on its device.
        unsafe { self.native_ref() }
    }
}
