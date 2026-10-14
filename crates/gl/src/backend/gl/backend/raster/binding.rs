//! Reflection-driven bind-group lowering for immediate GL raster encoders.
//!
//! A linked program chooses actual UBO points and combined-sampler units. This
//! module applies that immutable reflection directly; it never creates a
//! backend binding packet for a later draw. Dynamic offsets follow Core's
//! ascending logical-binding order and are checked before the native range is
//! installed.

use fluxel_rhi_core::api::{Error, Result};

use crate::backend::gl::backend::spi::native_error;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
use crate::backend::gl::backend::types::GlRasterPipeline;
use crate::backend::gl::backend::types::{GlBindingSlot, GlBoundResource, GlEncoder};

/// Validates Core's positional dynamic-offset ABI before changing GL state.
pub(crate) fn validate_dynamic_offsets(
    group: &super::super::types::GlBindGroup,
    dynamic_offsets: &[u32],
) -> Result<()> {
    let expected = group
        .layout
        .entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.slot,
                GlBindingSlot::UniformBuffer {
                    dynamic_offset: true,
                    ..
                }
            )
        })
        .count();
    if dynamic_offsets.len() != expected {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

fn dynamic_offset(
    group: &super::super::types::GlBindGroup,
    binding: u32,
    offsets: &[u32],
) -> Result<u32> {
    let index = group
        .layout
        .entries
        .iter()
        .filter(|entry| {
            entry.binding < binding
                && matches!(
                    entry.slot,
                    GlBindingSlot::UniformBuffer {
                        dynamic_offset: true,
                        ..
                    }
                )
        })
        .count();
    match group
        .layout
        .entries
        .iter()
        .find(|entry| entry.binding == binding)
        .map(|entry| entry.slot)
    {
        Some(GlBindingSlot::UniformBuffer {
            dynamic_offset: true,
            ..
        }) => offsets.get(index).copied().ok_or(Error::InvalidRange),
        Some(GlBindingSlot::UniformBuffer {
            dynamic_offset: false,
            ..
        }) => Ok(0),
        _ => Err(Error::InvalidState),
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(crate) fn bind_native_group(
    encoder: &mut GlEncoder,
    layout: &crate::backend::gl::backend::types::GlPipelineLayout,
    reflection: &crate::backend::gl::api::GlProgramReflection,
    index: u32,
    group: &super::super::types::GlBindGroup,
    offsets: &[u32],
) -> Result<()> {
    use crate::backend::gl::api::{
        GlBindingApi as _, GlExecutableBindingLocation, GlStorageImageApi as _, GlTextureTarget,
    };
    let result = encoder
        .state
        .with_native("bind reflected GL group", |provider| {
            for assignment in &reflection.assignments {
                if assignment.logical.group != index {
                    continue;
                }
                let binding = assignment.logical.binding;
                let entry = group
                    .layout
                    .entries
                    .iter()
                    .find(|entry| entry.binding == binding)
                    .ok_or(crate::backend::gl::api::GlError::Driver {
                        operation: "bind reflected GL group",
                        message: "linked binding is absent from group layout".into(),
                    })?;
                match (entry.slot, assignment.executable) {
                    (
                        GlBindingSlot::UniformBuffer { .. },
                        GlExecutableBindingLocation::UniformBlock(point),
                    ) => {
                        let Some((
                            _,
                            GlBoundResource::UniformBuffer {
                                buffer,
                                offset,
                                size,
                            },
                        )) = group.resources.iter().find(|(key, _)| *key == binding)
                        else {
                            return Err(crate::backend::gl::api::GlError::Driver {
                                operation: "bind reflected GL group",
                                message: "uniform resource is absent".into(),
                            });
                        };
                        let offset = offset
                            .checked_add(dynamic_offset(group, binding, offsets).map_err(|_| {
                                crate::backend::gl::api::GlError::Driver {
                                    operation: "bind reflected GL group",
                                    message: "invalid dynamic offset".into(),
                                }
                            })?)
                            .ok_or(crate::backend::gl::api::GlError::OutOfMemory {
                                operation: "bind reflected GL group",
                            })?;
                        provider.bind_uniform_buffer(point, Some(buffer.id), offset, *size)?;
                    }
                    (
                        GlBindingSlot::SampledTexture { .. },
                        GlExecutableBindingLocation::TextureUnit(unit),
                    ) => {
                        let Some((_, GlBoundResource::Texture(view))) =
                            group.resources.iter().find(|(key, _)| *key == binding)
                        else {
                            return Err(crate::backend::gl::api::GlError::Driver {
                                operation: "bind reflected GL group",
                                message: "sampled texture is absent".into(),
                            });
                        };
                        let pair = layout
                            .native
                            .bindings
                            .iter()
                            .find(|value| value.location == assignment.logical)
                            .and_then(|value| value.pair);
                        let sampler = if let Some(pair) = pair {
                            let Some((_, GlBoundResource::Sampler(sampler))) =
                                group.resources.iter().find(|(key, _)| *key == pair.binding)
                            else {
                                return Err(crate::backend::gl::api::GlError::Driver {
                                    operation: "bind reflected GL group",
                                    message: "paired sampler is absent".into(),
                                });
                            };
                            Some(sampler)
                        } else {
                            None
                        };
                        let target = match view.texture.desc.dimension {
                            crate::backend::gl::api::GlTextureDimension::D2 => GlTextureTarget::D2,
                            crate::backend::gl::api::GlTextureDimension::D2Array => {
                                GlTextureTarget::D2Array
                            }
                            _ => {
                                return Err(crate::backend::gl::api::GlError::Unsupported {
                                    operation: "bind reflected GL group",
                                    reason: "sampled texture dimension",
                                });
                            }
                        };
                        provider.active_texture(unit)?;
                        provider.bind_texture(unit, target, Some(view.texture.id))?;
                        // A WGSL `textureLoad` has no sampler resource. GLSL
                        // still expresses it through a sampler uniform, but
                        // `texelFetch` ignores filtering; clear any sampler
                        // object left on this unit so only the texture binding
                        // participates in the immediate operation.
                        provider.bind_sampler(unit, sampler.map(|value| value.id))?;
                    }
                    (
                        GlBindingSlot::StorageTexture { access, .. },
                        GlExecutableBindingLocation::ImageUnit(unit),
                    ) => {
                        let Some((_, GlBoundResource::StorageTexture(view))) =
                            group.resources.iter().find(|(key, _)| *key == binding)
                        else {
                            return Err(crate::backend::gl::api::GlError::Driver {
                                operation: "bind reflected GL group",
                                message: "storage texture is absent".into(),
                            });
                        };
                        let access = match access {
                            fluxel_rhi_core::api::StorageAccess::Read => {
                                crate::backend::gl::api::GlStorageImageAccess::ReadOnly
                            }
                            fluxel_rhi_core::api::StorageAccess::Write => {
                                crate::backend::gl::api::GlStorageImageAccess::WriteOnly
                            }
                            fluxel_rhi_core::api::StorageAccess::ReadWrite => {
                                crate::backend::gl::api::GlStorageImageAccess::ReadWrite
                            }
                        };
                        provider.bind_storage_image(
                            unit,
                            crate::backend::gl::api::GlStorageImageBinding {
                                texture: view.texture.id,
                                level: view.native.mip_level,
                                sample_count: view.native.sample_count,
                                layered: false,
                                layer: Some(view.native.array_layer),
                                format: view.native.format,
                                access,
                            },
                        )?;
                    }
                    (GlBindingSlot::Sampler, GlExecutableBindingLocation::TextureUnit(_)) => {}
                    _ => {
                        return Err(crate::backend::gl::api::GlError::Unsupported {
                            operation: "bind reflected GL group",
                            reason: "unimplemented reflected binding kind",
                        });
                    }
                }
            }
            Ok(())
        })
        .map_err(native_error);
    if result.is_ok() {
        encoder.issued = true;
    } else {
        encoder.indeterminate = true;
    }
    result
}

#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
pub(super) fn bind_browser_group(
    encoder: &mut GlEncoder,
    pipeline: &GlRasterPipeline,
    index: u32,
    group: &super::super::types::GlBindGroup,
    offsets: &[u32],
) -> Result<()> {
    use crate::backend::gl::api::GlExecutableBindingLocation;
    let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
    let result = (|| -> Result<()> {
        for assignment in &pipeline.reflection.assignments {
            if assignment.logical.group != index {
                continue;
            }
            let binding = assignment.logical.binding;
            let entry = group
                .layout
                .entries
                .iter()
                .find(|entry| entry.binding == binding)
                .ok_or(Error::InvalidState)?;
            let logical = pipeline
                .layout
                .native
                .bindings
                .iter()
                .find(|value| value.location == assignment.logical)
                .ok_or(Error::InvalidState)?;
            let name = logical.names.first().ok_or(Error::InvalidState)?;
            match (entry.slot, assignment.executable) {
                (
                    GlBindingSlot::UniformBuffer { .. },
                    GlExecutableBindingLocation::UniformBlock(point),
                ) => {
                    let Some((
                        _,
                        GlBoundResource::UniformBuffer {
                            buffer,
                            offset,
                            size,
                        },
                    )) = group.resources.iter().find(|(key, _)| *key == binding)
                    else {
                        return Err(Error::InvalidState);
                    };
                    let offset = offset
                        .checked_add(dynamic_offset(group, binding, offsets)?)
                        .ok_or(Error::InvalidRange)?;
                    let alignment = encoder
                        .state
                        .owner
                        .discovery_snapshot()
                        .limits()
                        .uniform_buffer_offset_alignment;
                    if alignment == 0 || u64::from(offset) % alignment != 0 {
                        return Err(Error::InvalidRange);
                    }
                    browser.bind_uniform_buffer_named(
                        &pipeline.browser_program,
                        name,
                        point,
                        &buffer.browser,
                        u64::from(offset),
                        u64::from(*size),
                    )?;
                }
                (
                    GlBindingSlot::SampledTexture { .. },
                    GlExecutableBindingLocation::TextureUnit(unit),
                ) => {
                    let Some((_, GlBoundResource::Texture(view))) =
                        group.resources.iter().find(|(key, _)| *key == binding)
                    else {
                        return Err(Error::InvalidState);
                    };
                    let pair = logical.pair.ok_or(Error::InvalidState)?;
                    let Some((_, GlBoundResource::Sampler(sampler))) =
                        group.resources.iter().find(|(key, _)| *key == pair.binding)
                    else {
                        return Err(Error::InvalidState);
                    };
                    browser.bind_rgba8_texture_named(
                        &pipeline.browser_program,
                        name,
                        unit,
                        &view.texture.browser,
                        &sampler.browser,
                    )?;
                }
                (GlBindingSlot::Sampler, GlExecutableBindingLocation::TextureUnit(_)) => {}
                _ => return Err(Error::Unsupported("WebGL2 reflected binding kind")),
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
