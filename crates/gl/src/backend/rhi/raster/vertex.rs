//! Immediate vertex and index buffer binding validation and lowering.
//!
//! This module keeps buffer provenance/range checks close to their native
//! bind operations; it stores only the resulting current binding state.

use super::*;

pub(super) fn set_vertex_buffer(
    encoder: &mut GlEncoder,
    slot: u32,
    buffer: &GlBuffer,
    offset: u64,
    size: u64,
) -> Result<()> {
    if offset
        .checked_add(size)
        .is_none_or(|end| end > buffer.desc.size)
    {
        return Err(Error::InvalidRange);
    }
    let pipeline = encoder
        .raster_pipeline
        .as_ref()
        .ok_or(Error::InvalidState)?;
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        let binding = crate::backend::api::GlVertexBufferBinding {
            slot,
            buffer: buffer.id,
            offset,
        };
        use crate::backend::api::GlVertexApi as _;
        let result = encoder
            .state
            .with_native("set GL vertex binding", |provider| {
                provider.set_vertex_binding(pipeline.vertex_array, binding)
            })
            .map_err(native_error);
        match result {
            Ok(()) => {
                encoder.issued = true;
                Ok(())
            }
            Err(error) => {
                encoder.indeterminate = true;
                Err(error)
            }
        }
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
            let layout = pipeline
                .vertex_layout
                .buffers
                .iter()
                .find(|layout| layout.slot == slot)
                .ok_or(Error::InvalidRange)?;
            let result = (|| -> Result<()> {
                browser.set_browser_vertex_array(&pipeline.browser_vertex_array)?;
                for attribute in pipeline
                    .vertex_layout
                    .attributes
                    .iter()
                    .filter(|attribute| attribute.buffer_slot == slot)
                {
                    browser.set_vertex_attribute(&buffer.browser, layout, attribute, offset)?;
                }
                Ok(())
            })();
            if result.is_ok() {
                encoder
                    .browser_vertex_bindings
                    .retain(|(old, _, _, _)| *old != slot);
                encoder
                    .browser_vertex_bindings
                    .push((slot, buffer.clone(), offset, size));
                encoder.issued = true;
            } else {
                encoder.indeterminate = true;
            }
            result
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        {
            let _ = (pipeline, slot, buffer, offset);
            Err(Error::Unsupported("GL platform context"))
        }
    }
}

pub(super) fn set_index_buffer(
    encoder: &mut GlEncoder,
    buffer: &GlBuffer,
    offset: u64,
    size: u64,
    format: IndexFormat,
) -> Result<()> {
    if offset
        .checked_add(size)
        .is_none_or(|end| end > buffer.desc.size)
    {
        return Err(Error::InvalidRange);
    }
    let pipeline = encoder
        .raster_pipeline
        .as_ref()
        .ok_or(Error::InvalidState)?;
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        let binding = crate::backend::api::GlIndexBinding {
            buffer: buffer.id,
            offset,
            format: match format {
                IndexFormat::Uint16 => crate::backend::api::GlIndexFormat::Uint16,
                IndexFormat::Uint32 => crate::backend::api::GlIndexFormat::Uint32,
            },
        };
        use crate::backend::api::GlVertexApi as _;
        let result = encoder
            .state
            .with_native("set GL index binding", |provider| {
                provider.set_index_binding(pipeline.vertex_array, Some(binding))
            })
            .map_err(native_error);
        match result {
            Ok(()) => {
                encoder.issued = true;
                Ok(())
            }
            Err(error) => {
                encoder.indeterminate = true;
                Err(error)
            }
        }
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            if offset > i32::MAX as u64 {
                return Err(Error::InvalidRange);
            }
            let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
            let result = (|| -> Result<()> {
                browser.set_browser_vertex_array(&pipeline.browser_vertex_array)?;
                browser.set_index_buffer(&buffer.browser)?;
                Ok(())
            })();
            if result.is_err() {
                encoder.indeterminate = true;
                return result;
            }
            encoder.browser_index = Some((offset, format));
            encoder.issued = true;
            Ok(())
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        {
            let _ = (pipeline, buffer, offset, format);
            Err(Error::Unsupported("GL platform context"))
        }
    }
}
