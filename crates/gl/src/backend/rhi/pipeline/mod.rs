//! Raster-pipeline creation for the immediate GL backend.
//!
//! Linking and VAO allocation happen on the cold creation path. The resulting
//! pipeline retains only native object identities and immutable fixed state;
//! draw calls are still issued directly by the encoder.

use std::future::{Ready, ready};

use fluxel_rhi_core::api::{self, Error, RasterPipelineDesc, Result, VertexFormat, VertexStepMode};
use fluxel_rhi_core::backend::PipelineBackend;

use super::spi::native_error;
use super::types::{GlPipelineLayout, GlRasterPipeline, GlRasterPipelineInner};
use super::{Gl, GlDevice};

mod bindings;
mod fixed_state;
pub(crate) use bindings::{validate_naga_resource_types, with_naga_binding_names};
use fixed_state::{lower_depth_stencil, lower_vertex_layout};
mod limits;
mod vertex_format;
use limits::{
    validate_combined_buffer_limits, validate_combined_image_limits,
    validate_combined_sampled_limits,
};
use vertex_format::lower_vertex_format;

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
        crate::backend::translate_raster::lower_primitive(desc.primitive)?;
    let depth_stencil = desc.depth_stencil.map(lower_depth_stencil).transpose()?;
    let rasterization = crate::backend::api::GlRasterizationState {
        polygon_mode: match desc.primitive.polygon_mode {
            api::PolygonMode::Fill => crate::backend::api::GlPolygonMode::Fill,
            api::PolygonMode::Line => crate::backend::api::GlPolygonMode::Line,
            api::PolygonMode::Point => crate::backend::api::GlPolygonMode::Point,
        },
        depth_clamp: desc.primitive.unclipped_depth,
        conservative: desc.primitive.conservative,
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    if !device
        .0
        .with_native("validate GL rasterization modes", |provider| {
            Ok(provider.supports_rasterization_topology(rasterization, topology))
        })
        .map_err(native_error)?
    {
        return Err(Error::Unsupported("native GL/GLES rasterization modes"));
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        use crate::backend::api::{GlKnownExtension, GlPolygonMode};
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
            Some(target) => Ok(crate::backend::api::GlColorTargetState {
                write_mask: target.write_mask,
                blend: target
                    .blend
                    .map(|blend| {
                        Ok(crate::backend::api::GlBlendState {
                            color: crate::backend::translate_raster::lower_blend_component(
                                blend.color,
                                false,
                            )?,
                            alpha: crate::backend::translate_raster::lower_blend_component(
                                blend.alpha,
                                true,
                            )?,
                        })
                    })
                    .transpose()?,
            }),
            None => Ok(crate::backend::api::GlColorTargetState {
                write_mask: 0,
                blend: None,
            }),
        })
        .collect::<Result<Vec<_>>>()?;
    let vertex_layout = lower_vertex_layout(&desc.vertex_buffers)?;
    let multisample = crate::backend::translate_raster::lower_multisample(desc.multisample)?;
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
                .is_acquired(crate::backend::api::GlKnownExtension::ExtPolygonOffsetClamp)
        {
            return Err(Error::Unsupported("WebGL2 polygon offset clamp"));
        }
    }
    // SAFETY: Core verified shader/layout provenance before calling this SPI.
    // We only read immutable backend source/layout descriptors while those
    // public handles are borrowed for this synchronous creation call.
    let layout = unsafe { desc.layout.native_ref() }.clone();
    let vertex = super::shader::materialize_stage(
        device,
        unsafe { desc.vertex.module.native_ref() },
        &desc.vertex.entry,
        api::ShaderStage::Vertex,
        &desc.vertex.constants,
        desc.multiview.and_then(std::num::NonZeroU32::new),
        &layout,
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
                &layout,
            )
        })
        .transpose()?;
    let vertex = &vertex;
    let fragment = fragment.as_ref();
    // SAFETY: Core retains the pipeline-layout handle for synchronous
    // pipeline creation. Copy its immutable ABI before native linking so the
    // returned pipeline can reject incompatible bind groups later.
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
            source.stage = crate::backend::api::GlShaderStage::Fragment;
            source.entry_point = "main".into();
            source.source_hash = crate::backend::api::GlShaderKey([0x69; 32]);
            if source.spirv.is_some() {
                source.spirv = Some(super::shader::spirv::empty_fragment());
                source.source_hash = crate::backend::api::GlShaderKey([0x6a; 32]);
                source.text.clear();
                return source;
            }
            source.text = match source.dialect {
                crate::backend::api::GlShaderDialect::Desktop { version } => {
                    format!("#version {version} core\nvoid main() {{}}\n")
                }
                crate::backend::api::GlShaderDialect::Embedded { version } => {
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
    let descriptor = crate::backend::api::GlProgramDescriptor {
        kind: crate::backend::api::GlProgramKind::Raster {
            vertex: vertex_source,
            fragment: fragment_source,
        },
        layout: native_layout,
        debug_name: desc.label.clone(),
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::api::{GlShaderApi as _, GlVertexApi as _};
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
                        Ok(false) => { let _ = provider.destroy_program(program); return Err(crate::backend::api::GlError::Unsupported { operation: "dual-source blending", reason: "linked fragment shader has no secondary output at location zero" }); },
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
                    raster: crate::backend::api::GlRasterState {
                        rasterization,
                        topology,
                        cull_mode,
                        front_face,
                        depth_stencil,
                        color_targets,
                        multisample,
                        // The active pass supplies its validated drawable size
                        // at `set_pipeline`; creation cannot guess it.
                        viewport: crate::backend::api::GlViewport {
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
                raster: crate::backend::api::GlRasterState {
                    rasterization,
                    topology,
                    cull_mode,
                    front_face,
                    depth_stencil,
                    color_targets,
                    multisample,
                    viewport: crate::backend::api::GlViewport {
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

fn harmonize_embedded_versions(
    vertex: &mut crate::backend::api::GlShaderSource,
    fragment: &mut crate::backend::api::GlShaderSource,
) -> Result<()> {
    use crate::backend::api::GlShaderDialect;
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
        source.source_hash = crate::backend::api::GlShaderKey(super::shader::source_key(
            source.text.as_bytes(),
            &source.entry_point,
        ));
    }
    Ok(())
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
fn validate_webgl2_fixed_state(
    colors: &[crate::backend::api::GlColorTargetState],
    depth: Option<crate::backend::api::GlDepthStencilState>,
    multisample: crate::backend::api::GlMultisampleState,
) -> Result<()> {
    let _ = (colors, depth);
    if multisample.sample_mask != u64::MAX {
        return Err(Error::Unsupported("WebGL2 sample mask"));
    }
    Ok(())
}

/// Image qualifiers are emitted before the other shader stage is known. Assign
/// one dense unit per logical binding at link time and rewrite only Naga's
/// corresponding image declarations in each source.
fn remap_raster_image_units(
    layout: &mut crate::backend::api::GlPipelineLayout,
    vertex: &super::types::GlShader,
    vertex_source: &mut crate::backend::api::GlShaderSource,
    fragment: Option<&super::types::GlShader>,
    fragment_source: &mut crate::backend::api::GlShaderSource,
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
        source.source_hash = crate::backend::api::GlShaderKey(super::shader::source_key(
            source.text.as_bytes(),
            &source.entry_point,
        ));
    }
    Ok(())
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
