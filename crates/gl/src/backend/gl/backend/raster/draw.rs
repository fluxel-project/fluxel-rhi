//! Direct draw lowering for GL raster passes.
//!
//! The immediate encoder retains only current VAO/index state. Each call here
//! issues a native draw immediately and cannot be replayed by `submit`.

use fluxel_rhi_core::api::{Error, IndexFormat, Result};

use crate::backend::gl::backend::spi::native_error;
use crate::backend::gl::backend::types::{GlEncoder, GlRasterPipeline};

pub(super) fn issue_draw(
    encoder: &mut GlEncoder,
    pipeline: &GlRasterPipeline,
    draw: crate::backend::gl::api::GlDrawCommand,
) -> Result<()> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::api::GlRasterCommandApi as _;
        let result = encoder
            .state
            .with_native("draw GL raster", |provider| provider.draw_raster(draw))
            .map_err(native_error);
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
            use crate::backend::gl::api::{GlDrawCommand, GlPrimitiveTopology};
            use web_sys::WebGl2RenderingContext as WebGl;
            let mode = match pipeline.raster.topology {
                GlPrimitiveTopology::Points => WebGl::POINTS,
                GlPrimitiveTopology::Lines => WebGl::LINES,
                GlPrimitiveTopology::LineStrip => WebGl::LINE_STRIP,
                GlPrimitiveTopology::Triangles => WebGl::TRIANGLES,
                GlPrimitiveTopology::TriangleStrip => WebGl::TRIANGLE_STRIP,
            };
            let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
            let result = (|| -> Result<()> {
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
                        browser.draw_elements(
                            mode,
                            i32::try_from(draw.index_count).map_err(|_| Error::InvalidRange)?,
                            kind,
                            i32::try_from(offset).map_err(|_| Error::InvalidRange)?,
                            i32::try_from(draw.instance_count).map_err(|_| Error::InvalidRange)?,
                        )?;
                    }
                }
                Ok(())
            })();
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
