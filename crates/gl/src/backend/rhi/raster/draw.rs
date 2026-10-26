//! Direct draw lowering for GL raster passes.
//!
//! The immediate encoder retains only current VAO/index state. Each call here
//! issues a native draw immediately and cannot be replayed by `submit`.

use fluxel_rhi_core::api::{Error, IndexFormat, Result};

use crate::backend::rhi::spi::native_error;
use crate::backend::rhi::types::{GlEncoder, GlRasterPipeline};

/// WebGL validates the secondary fragment output for dual-source blending at
/// draw time. Keep that narrow contract synchronous: unlike ordinary draw
/// errors it is a pipeline/program compatibility failure the caller has
/// historically received from `draw`, and delaying it to `end_raster` would
/// make an invalid pipeline appear accepted. All other draw errors aggregate at
/// the raster-pass boundary.
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
fn browser_dual_source_draw_boundary(pipeline: &GlRasterPipeline) -> bool {
    use crate::backend::api::GlBlendFactor;
    pipeline.raster.color_targets.iter().any(|target| {
        target.blend.is_some_and(|blend| {
            [
                blend.color.src_factor,
                blend.color.dst_factor,
                blend.alpha.src_factor,
                blend.alpha.dst_factor,
            ]
            .iter()
            .any(|factor| {
                matches!(
                    factor,
                    GlBlendFactor::Src1
                        | GlBlendFactor::OneMinusSrc1
                        | GlBlendFactor::Src1Alpha
                        | GlBlendFactor::OneMinusSrc1Alpha
                )
            })
        })
    })
}

pub(super) fn issue_draw(
    encoder: &mut GlEncoder,
    pipeline: &GlRasterPipeline,
    draw: crate::backend::api::GlDrawCommand,
    base_vertex: i32,
    first_instance: u32,
) -> Result<()> {
    super::cube_mirror::refresh(encoder)?;
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::api::GlRasterCommandApi as _;
        let result = encoder
            .state
            .with_native("draw GL raster", |provider| {
                provider.draw_direct_advanced(draw, base_vertex, first_instance)
            })
            .map_err(|error| match error {
                crate::backend::api::GlError::Unsupported { reason, .. } => {
                    Error::Unsupported(reason)
                }
                other => native_error(other),
            });
        if result.is_ok() {
            encoder.issued = true;
        } else {
            encoder.indeterminate = true;
        }
        result
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            use crate::backend::api::{GlDrawCommand, GlPrimitiveTopology};
            use web_sys::WebGl2RenderingContext as WebGl;
            let mode = match pipeline.raster.topology {
                GlPrimitiveTopology::Points => WebGl::POINTS,
                GlPrimitiveTopology::Lines => WebGl::LINES,
                GlPrimitiveTopology::LineStrip => WebGl::LINE_STRIP,
                GlPrimitiveTopology::Triangles => WebGl::TRIANGLES,
                GlPrimitiveTopology::TriangleStrip => WebGl::TRIANGLE_STRIP,
            };
            let mut shifted = Vec::new();
            if first_instance != 0 {
                for layout in pipeline.vertex_layout.buffers.iter().filter(|l| {
                    l.step_mode == crate::backend::api::GlVertexStepMode::Instance
                        && l.stride != 0
                }) {
                    let (_, buffer, offset, _) = encoder
                        .browser_vertex_bindings
                        .iter()
                        .find(|(slot, _, _, _)| *slot == layout.slot)
                        .ok_or(Error::InvalidState)?;
                    for attribute in pipeline
                        .vertex_layout
                        .attributes
                        .iter()
                        .filter(|a| a.buffer_slot == layout.slot)
                    {
                        let stride = layout.stride;
                        let target = offset
                            .checked_add(u64::from(first_instance) * u64::from(stride))
                            .ok_or(Error::InvalidRange)?;
                        if target > buffer.desc.size {
                            return Err(Error::InvalidRange);
                        }
                        shifted.push((*layout, *attribute, buffer.clone(), *offset, target));
                    }
                }
            }
            let mut constants = Vec::new();
            for layout in pipeline.vertex_layout.buffers.iter().filter(|layout| {
                layout.stride == 0
                    && pipeline
                        .vertex_layout
                        .attributes
                        .iter()
                        .any(|a| a.buffer_slot == layout.slot)
            }) {
                let (_, buffer, offset, size) = encoder
                    .browser_vertex_bindings
                    .iter()
                    .find(|(slot, _, _, _)| *slot == layout.slot)
                    .ok_or(Error::InvalidState)?;
                for attribute in pipeline
                    .vertex_layout
                    .attributes
                    .iter()
                    .filter(|a| a.buffer_slot == layout.slot)
                {
                    constants.push((*attribute, buffer.clone(), *offset, *size));
                }
            }
            let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
            let strict_dual_source = browser_dual_source_draw_boundary(pipeline);
            let result = (|| -> Result<()> {
                use crate::backend::browser::BrowserPushUniformKind;
                browser.upload_push_uniform(
                    &pipeline.browser_program,
                    naga::back::glsl::FIRST_INSTANCE_BINDING,
                    BrowserPushUniformKind::Uint { lanes: 1 },
                    &[first_instance],
                )?;
                for (attribute, buffer, offset, size) in &constants {
                    browser.refresh_constant_vertex_attribute(
                        &buffer.browser,
                        attribute,
                        *offset,
                        *size,
                    )?;
                }
                for (layout, attribute, buffer, _, target) in &shifted {
                    browser.set_vertex_attribute(&buffer.browser, layout, attribute, *target)?;
                }
                match draw {
                    GlDrawCommand::NonIndexed(draw) => browser.draw_arrays(
                        mode,
                        i32::try_from(draw.first_vertex).map_err(|_| Error::InvalidRange)?,
                        i32::try_from(draw.vertex_count).map_err(|_| Error::InvalidRange)?,
                        i32::try_from(draw.instance_count).map_err(|_| Error::InvalidRange)?,
                    )?,
                    GlDrawCommand::Indexed(draw) => {
                        let (base, format) = encoder.browser_index.ok_or(Error::InvalidState)?;
                        let element_size = match format {
                            IndexFormat::Uint16 => 2_u64,
                            IndexFormat::Uint32 => 4_u64,
                        };
                        let offset = base
                            .checked_add(
                                u64::from(draw.first_index)
                                    .checked_mul(element_size)
                                    .ok_or(Error::InvalidRange)?,
                            )
                            .ok_or(Error::InvalidRange)?;
                        let kind = match format {
                            IndexFormat::Uint16 => WebGl::UNSIGNED_SHORT,
                            IndexFormat::Uint32 => WebGl::UNSIGNED_INT,
                        };
                        if base_vertex != 0 {
                            browser.draw_elements_base_vertex(
                                mode,
                                i32::try_from(draw.index_count).map_err(|_| Error::InvalidRange)?,
                                kind,
                                i32::try_from(offset).map_err(|_| Error::InvalidRange)?,
                                i32::try_from(draw.instance_count)
                                    .map_err(|_| Error::InvalidRange)?,
                                base_vertex,
                            )?;
                        } else {
                            browser.draw_elements(
                                mode,
                                i32::try_from(draw.index_count).map_err(|_| Error::InvalidRange)?,
                                kind,
                                i32::try_from(offset).map_err(|_| Error::InvalidRange)?,
                                i32::try_from(draw.instance_count)
                                    .map_err(|_| Error::InvalidRange)?,
                            )?;
                        }
                    }
                }
                if strict_dual_source {
                    browser.observe_raster_boundary("draw dual-source WebGL raster")?;
                }
                Ok(())
            })();
            let restore = (|| -> Result<()> {
                for (layout, attribute, buffer, offset, _) in &shifted {
                    browser.set_vertex_attribute(&buffer.browser, layout, attribute, *offset)?;
                }
                Ok(())
            })();
            let result = result.and(restore);
            if result.is_ok() {
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            result
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        {
            let _ = (encoder, pipeline, draw);
            Err(Error::Unsupported("GL platform context"))
        }
    }
}
