//! Consecutive OVR layer intervals preserve Core's logical view indices.
use super::*;

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
mod browser;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
mod native;

const BASE: &str = "fluxel_internal_view_index_base";

pub(super) fn begin(encoder: &mut GlEncoder, desc: &RasterPassDesc<'_, Gl>) -> Result<()> {
    let mask = desc.multiview_mask.ok_or(Error::InvalidState)?;
    if mask == 0 {
        return Err(Error::InvalidRange);
    }
    let first = mask.trailing_zeros();
    let count = mask.count_ones();
    if mask >> first != u32::MAX >> (32 - count) {
        return Err(Error::Unsupported("OVR noncontiguous multiview mask"));
    }
    let limit = super::super::instance::multiview_limit(&encoder.state)?;
    if limit == 0 {
        return Err(Error::Unsupported("GL OVR multiview"));
    }
    if count > limit {
        return Err(Error::InvalidRange);
    }
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        return native::begin(encoder, desc, first, count);
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        return browser::begin(encoder, desc, first, count);
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = (first, count);
        Err(Error::Unsupported("GL multiview platform"))
    }
}

pub(super) fn install(encoder: &mut GlEncoder, pipeline: &GlRasterPipeline) -> Result<()> {
    if pipeline.multiview != encoder.raster_multiview.map(|(count, _)| count) {
        return Err(Error::InvalidState);
    }
    let Some((_, first)) = encoder.raster_multiview else {
        return Ok(());
    };
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::gl::native::push_uniform::GlPushUniformValue;
        let result = encoder
            .state
            .with_native("set multiview base", |provider| {
                provider.set_push_uniforms(
                    pipeline.program,
                    &[(BASE.into(), GlPushUniformValue::Sint(vec![first as i32]))],
                )
            })
            .map_err(native_error);
        if result.is_err() {
            encoder.indeterminate = true;
        }
        result?;
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let result = encoder
            .browser
            .as_mut()
            .ok_or(Error::InvalidState)?
            .upload_push_uniform(
                &pipeline.browser_program,
                BASE,
                crate::backend::gl::browser::BrowserPushUniformKind::Sint { lanes: 1 },
                &[first],
            );
        if result.is_err() {
            encoder.indeterminate = true;
        }
        result?;
    }
    Ok(())
}
