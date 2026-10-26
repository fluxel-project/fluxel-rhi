//! Refresh fallback cube sampling storage immediately before GPU execution.
//!
//! The source allocation remains authoritative. Copies precede every draw or
//! dispatch so writes made after binding are observed; this is not a snapshot
//! created at view construction. Concurrent writes to the source are rejected.
use crate::backend::rhi::types::{GlBoundResource, GlEncoder, GlTextureView};
use fluxel_rhi_core::api::{Error, Result};

fn visit(
    resource: &GlBoundResource,
    writable: bool,
    sampled: &mut Vec<GlTextureView>,
    storage: &mut Vec<GlTextureView>,
) {
    match resource {
        GlBoundResource::Texture(view) => sampled.push(view.clone()),
        GlBoundResource::StorageTexture(view) if writable => storage.push(view.clone()),
        GlBoundResource::Array(elements) => {
            for item in elements.iter().flatten() {
                visit(item, writable, sampled, storage);
            }
        }
        _ => {}
    }
}

/// Returns whether a native/browser utility pass changed GL state.  Native
/// callers must reapply their CPU-owned pipeline packet before issuing the
/// following draw or dispatch.
pub(crate) fn refresh(encoder: &mut GlEncoder) -> Result<bool> {
    let (layout, reflection) = if let Some(pipeline) = &encoder.raster_pipeline {
        (&pipeline.layout, &pipeline.reflection)
    } else if let Some(pipeline) = &encoder.compute_pipeline {
        (&pipeline.layout, &pipeline.reflection)
    } else {
        return Err(Error::InvalidState);
    };
    let groups = super::binding::compatible_groups(encoder, layout);
    let mut sampled = Vec::new();
    let mut storage = Vec::new();
    for (_, group, _) in &groups {
        for (binding, resource) in group.resources.iter() {
            let writable = group.layout.entries.iter().any(|entry| entry.binding == *binding && matches!(entry.slot,
                crate::backend::rhi::types::GlBindingSlot::StorageTexture { access, .. } if access != fluxel_rhi_core::api::StorageAccess::Read));
            visit(resource, writable, &mut sampled, &mut storage);
        }
    }
    let mut active_sampled = Vec::new();
    for assignment in &reflection.assignments {
        if !matches!(
            assignment.executable,
            crate::backend::api::GlExecutableBindingLocation::TextureUnit(_)
        ) {
            continue;
        }
        if let Some((_, group, _)) = groups
            .iter()
            .find(|(slot, _, _)| *slot == assignment.logical.group)
        {
            if let Some((_, GlBoundResource::Texture(view))) = super::binding::reflected_resource(
                group,
                assignment.logical.binding,
                assignment.element,
            ) {
                let crate::backend::api::GlExecutableBindingLocation::TextureUnit(unit) =
                    assignment.executable
                else {
                    unreachable!()
                };
                active_sampled.push((view.clone(), unit));
            }
        }
    }
    for (index, (view, _)) in active_sampled.iter().enumerate() {
        if !matches!(
            view.binding_texture().desc.format,
            crate::backend::api::GlFormat::Depth24PlusStencil8
                | crate::backend::api::GlFormat::Depth32FloatStencil8
        ) {
            continue;
        }
        let stencil = view.aspects == fluxel_rhi_core::api::TextureAspects::STENCIL;
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        if view.stencil_mirror.is_some() {
            continue;
        }
        if active_sampled[index + 1..].iter().any(|(other, _)| {
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            if other.stencil_mirror.is_some() {
                return false;
            }
            std::ptr::eq(&**view.binding_texture(), &**other.binding_texture())
                && stencil != (other.aspects == fluxel_rhi_core::api::TextureAspects::STENCIL)
        }) {
            return Err(Error::Unsupported(
                "GL simultaneous depth and stencil sampling from one texture object",
            ));
        }
    }
    let mut refreshed = false;
    for (view, unit) in active_sampled {
        #[cfg(any(
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl",
            all(target_arch = "wasm32", feature = "webgl2")
        ))]
        if let Some(mirror) = &view.stencil_mirror {
            if encoder
                .raster_views
                .iter()
                .chain(storage.iter())
                .any(|other| std::ptr::eq(&*view.texture, &*other.texture))
            {
                return Err(Error::Unsupported(
                    "GL stencil mirror concurrent source write",
                ));
            }
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            let result = encoder
                .browser
                .as_mut()
                .ok_or(Error::InvalidState)?
                .refresh_stencil_mirror(
                    &view.texture.browser,
                    mirror,
                    view.source_mip,
                    view.source_layer,
                );
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            let result = encoder
                .state
                .with_native("refresh stencil mirror", |provider| {
                    provider.sync_stencil_mirror(
                        view.texture.id,
                        mirror.id,
                        view.source_mip,
                        view.source_layer,
                    )
                })
                .map_err(crate::backend::rhi::spi::native_error);
            if let Err(error) = result {
                encoder.indeterminate = true;
                return Err(error);
            }
            encoder.issued = true;
            refreshed = true;
        }
        #[cfg(any(
            all(windows, feature = "native-gl-wgl"),
            feature = "native-gles-egl",
            all(target_arch = "wasm32", feature = "webgl2")
        ))]
        if let Some(mirror) = &view.cube_mirror {
            if encoder
                .raster_views
                .iter()
                .chain(storage.iter())
                .any(|other| std::ptr::eq(&*view.texture, &*other.texture))
            {
                return Err(Error::Unsupported("GL cube mirror concurrent source write"));
            }
            #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
            let result = encoder
                .browser
                .as_mut()
                .ok_or(Error::InvalidState)?
                .refresh_cube_mirror(
                    &view.texture.browser,
                    mirror,
                    view.source_mip,
                    view.mip_count,
                    view.source_layer,
                    unit,
                );
            #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
            let result = encoder
                .state
                .with_native("refresh cube mirror", |provider| {
                    provider.sync_cube_mirror(
                        view.texture.id,
                        mirror.id,
                        view.source_mip,
                        view.source_layer,
                    )
                })
                .map_err(crate::backend::rhi::spi::native_error);
            if let Err(error) = result {
                encoder.indeterminate = true;
                return Err(error);
            }
            encoder.issued = true;
            refreshed = true;
        }
    }
    Ok(refreshed)
}
