//! Reflection-driven bind-group lowering for immediate GL raster encoders.
//!
//! A linked program chooses actual UBO points and combined-sampler units. This
//! module applies that immutable reflection directly; it never creates a
//! backend binding packet for a later draw. Dynamic offsets follow Core's
//! ascending logical-binding order and are checked before the native range is
//! installed.

use fluxel_rhi_core::api::{Error, Result};

use crate::backend::rhi::spi::native_error;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
use crate::backend::rhi::types::GlRasterPipeline;
use crate::backend::rhi::types::{GlBindingSlot, GlBoundResource, GlEncoder};

// Runtime arrays may omit a suffix even when holes are forbidden by Core.
// This only selects the private native dummy binding; it does not relax Core
// validation or permit shader accesses to absent logical elements.
fn allows_missing_element(entry: &super::super::types::GlBindingLayoutEntry) -> bool {
    entry.partially_bound
        || matches!(
            entry.count,
            fluxel_rhi_core::api::BindingCount::RuntimeSized { .. }
        )
}

pub(crate) fn reflected_resource(
    group: &super::super::types::GlBindGroup,
    binding: u32,
    element: Option<u32>,
) -> Option<(u32, &GlBoundResource)> {
    let (_, resource) = group.resources.iter().find(|(key, _)| *key == binding)?;
    let resource = match (resource, element) {
        (GlBoundResource::Array(elements), Some(index)) => {
            elements.get(index as usize)?.as_ref()?
        }
        (GlBoundResource::Array(_), None) | (_, Some(_)) => return None,
        (resource, None) => resource,
    };
    Some((binding, resource))
}

fn mip_count_for_binding(
    group: &super::super::types::GlBindGroup,
    uniform: &super::super::types::GlMipUniform,
) -> Result<u32> {
    match reflected_resource(group, uniform.binding, uniform.element) {
        Some((_, GlBoundResource::Texture(view))) => Ok(if uniform.samples {
            view.texture.desc.sample_count
        } else {
            view.mip_count
        }),
        None if uniform.element.is_some()
            && group
                .layout
                .entries
                .iter()
                .any(|entry| entry.binding == uniform.binding && allows_missing_element(entry)) =>
        {
            Ok(1)
        }
        _ => Err(Error::InvalidState),
    }
}

fn remember_group(
    encoder: &mut GlEncoder,
    index: u32,
    group: &super::super::types::GlBindGroup,
    offsets: &[u32],
) {
    encoder.bound_groups.retain(|(slot, _, _)| *slot != index);
    encoder
        .bound_groups
        .push((index, group.clone(), offsets.to_vec()));
}

pub(crate) fn compatible_groups(
    encoder: &GlEncoder,
    layout: &super::super::types::GlPipelineLayout,
) -> Vec<(u32, super::super::types::GlBindGroup, Vec<u32>)> {
    encoder
        .bound_groups
        .iter()
        .filter(|(slot, group, _)| {
            layout
                .groups
                .get(*slot as usize)
                .is_some_and(|expected| super::layout::matches(expected, &group.layout))
        })
        .cloned()
        .collect()
}

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
                } | GlBindingSlot::StorageBuffer {
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
                    } | GlBindingSlot::StorageBuffer {
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
        Some(
            GlBindingSlot::UniformBuffer {
                dynamic_offset: true,
                ..
            }
            | GlBindingSlot::StorageBuffer {
                dynamic_offset: true,
                ..
            },
        ) => offsets.get(index).copied().ok_or(Error::InvalidRange),
        Some(
            GlBindingSlot::UniformBuffer {
                dynamic_offset: false,
                ..
            }
            | GlBindingSlot::StorageBuffer {
                dynamic_offset: false,
                ..
            },
        ) => Ok(0),
        _ => Err(Error::InvalidState),
    }
}

#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
pub(crate) fn bind_native_group(
    encoder: &mut GlEncoder,
    layout: &crate::backend::rhi::types::GlPipelineLayout,
    reflection: &crate::backend::api::GlProgramReflection,
    program: crate::backend::api::ProgramId,
    mip_uniforms: &[super::super::types::GlMipUniform],
    index: u32,
    group: &super::super::types::GlBindGroup,
    offsets: &[u32],
) -> Result<()> {
    use crate::backend::api::{
        GlBindingApi as _, GlExecutableBindingLocation, GlStorageBufferApi as _,
        GlStorageImageApi as _, GlTextureTarget,
    };
    let mip_counts = mip_uniforms
        .iter()
        .filter(|uniform| uniform.group == index)
        .map(|uniform| Ok((&uniform.name, mip_count_for_binding(group, uniform)?)))
        .collect::<Result<Vec<_>>>()?;
    let result = encoder
        .state
        .with_native("bind reflected GL group", |provider| {
            for (name, count) in &mip_counts {
                provider.set_texture_mip_count(program, name, *count)?;
            }
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
                    .ok_or(crate::backend::api::GlError::Driver {
                        operation: "bind reflected GL group",
                        message: "linked binding is absent from group layout".into(),
                    })?;
                match (entry.slot, assignment.executable) {
                    (
                        GlBindingSlot::UniformBuffer { .. },
                        GlExecutableBindingLocation::UniformBlock(point),
                    ) => {
                        if reflected_resource(group, binding, assignment.element).is_none()
                            && allows_missing_element(entry)
                            && assignment.element.is_some()
                        {
                            provider.bind_dummy_uniform_buffer(program, point)?;
                            continue;
                        }
                        let Some((
                            _,
                            GlBoundResource::UniformBuffer {
                                buffer,
                                offset,
                                size,
                            },
                        )) = reflected_resource(group, binding, assignment.element)
                        else {
                            return Err(crate::backend::api::GlError::Driver {
                                operation: "bind reflected GL group",
                                message: "uniform resource is absent".into(),
                            });
                        };
                        let offset = offset
                            .checked_add(u64::from(
                                dynamic_offset(group, binding, offsets).map_err(|_| {
                                    crate::backend::api::GlError::Driver {
                                        operation: "bind reflected GL group",
                                        message: "invalid dynamic offset".into(),
                                    }
                                })?,
                            ))
                            .ok_or(crate::backend::api::GlError::OutOfMemory {
                                operation: "bind reflected GL group",
                            })?;
                        provider.bind_uniform_buffer(point, Some(buffer.id), offset, *size)?;
                    }
                    (
                        GlBindingSlot::StorageBuffer { access, .. },
                        GlExecutableBindingLocation::StorageBlock(point),
                    ) => {
                        if reflected_resource(group, binding, assignment.element).is_none()
                            && allows_missing_element(entry)
                            && assignment.element.is_some()
                        {
                            provider.bind_dummy_storage_buffer(program, point)?;
                            continue;
                        }
                        let Some((
                            _,
                            GlBoundResource::StorageBuffer {
                                buffer,
                                offset,
                                size,
                            },
                        )) = reflected_resource(group, binding, assignment.element)
                        else {
                            return Err(crate::backend::api::GlError::Driver {
                                operation: "bind reflected GL group",
                                message: "storage buffer is absent".into(),
                            });
                        };
                        let dynamic = dynamic_offset(group, binding, offsets).map_err(|_| {
                            crate::backend::api::GlError::Driver {
                                operation: "bind reflected GL group",
                                message: "invalid dynamic offset".into(),
                            }
                        })?;
                        let usage = match access {
                            fluxel_rhi_core::api::StorageAccess::Read => {
                                crate::backend::api::GlStorageBufferUsage::ReadOnly
                            }
                            _ => crate::backend::api::GlStorageBufferUsage::ReadWrite,
                        };
                        provider.bind_storage_buffer(
                            point,
                            crate::backend::api::GlStorageBufferRange {
                                buffer: buffer.id,
                                offset: offset.checked_add(u64::from(dynamic)).ok_or(
                                    crate::backend::api::GlError::OutOfMemory {
                                        operation: "bind reflected GL group",
                                    },
                                )?,
                                size: *size,
                                usage,
                            },
                        )?;
                    }
                    (
                        GlBindingSlot::SampledTexture {
                            dimension,
                            sample,
                            multisampled,
                        },
                        GlExecutableBindingLocation::TextureUnit(unit),
                    ) => {
                        let resource = reflected_resource(group, binding, assignment.element);
                        if resource.is_none()
                            && allows_missing_element(entry)
                            && assignment.element.is_some()
                        {
                            let target = match (dimension, multisampled) {
                                (fluxel_rhi_core::api::TextureViewDimension::D1, false) => {
                                    GlTextureTarget::D1
                                }
                                (fluxel_rhi_core::api::TextureViewDimension::D2, false) => {
                                    GlTextureTarget::D2
                                }
                                (fluxel_rhi_core::api::TextureViewDimension::D2, true) => {
                                    GlTextureTarget::D2Multisample
                                }
                                (fluxel_rhi_core::api::TextureViewDimension::D2Array, false) => {
                                    GlTextureTarget::D2Array
                                }
                                (fluxel_rhi_core::api::TextureViewDimension::D2Array, true) => {
                                    GlTextureTarget::D2MultisampleArray
                                }
                                (fluxel_rhi_core::api::TextureViewDimension::D3, false) => {
                                    GlTextureTarget::D3
                                }
                                (fluxel_rhi_core::api::TextureViewDimension::Cube, false) => {
                                    GlTextureTarget::Cube
                                }
                                (fluxel_rhi_core::api::TextureViewDimension::CubeArray, false) => {
                                    GlTextureTarget::CubeArray
                                }
                                _ => {
                                    return Err(crate::backend::api::GlError::Unsupported {
                                        operation: "bind reflected GL group",
                                        reason: "dummy sampled texture dimension",
                                    });
                                }
                            };
                            provider.bind_dummy_sampled_texture(unit, target, sample)?;
                            let has_sampler = reflection.assignments.iter().any(|other| {
                                other.executable == assignment.executable
                                    && layout.native.bindings.iter().any(|logical| {
                                        logical.location == other.logical
                                            && matches!(
                                                logical.kind,
                                                crate::backend::api::GlShaderResourceKind::Sampler
                                            )
                                    })
                            });
                            if !has_sampler {
                                provider.bind_sampler(unit, None)?;
                            }
                            continue;
                        }
                        let Some((_, GlBoundResource::Texture(view))) = resource else {
                            return Err(crate::backend::api::GlError::Driver {
                                operation: "bind reflected GL group",
                                message: "sampled texture is absent".into(),
                            });
                        };
                        let texture_desc = view.binding_texture().desc;
                        let target = match (texture_desc.dimension, texture_desc.sample_count > 1) {
                            (crate::backend::api::GlTextureDimension::D1, false) => {
                                GlTextureTarget::D1
                            }
                            (crate::backend::api::GlTextureDimension::D2, false) => {
                                GlTextureTarget::D2
                            }
                            (crate::backend::api::GlTextureDimension::D2, true) => {
                                GlTextureTarget::D2Multisample
                            }
                            (crate::backend::api::GlTextureDimension::D2Array, true) => {
                                GlTextureTarget::D2MultisampleArray
                            }
                            (crate::backend::api::GlTextureDimension::D2Array, false) => {
                                GlTextureTarget::D2Array
                            }
                            (crate::backend::api::GlTextureDimension::D3, false) => {
                                GlTextureTarget::D3
                            }
                            (crate::backend::api::GlTextureDimension::Cube, false) => {
                                GlTextureTarget::Cube
                            }
                            (crate::backend::api::GlTextureDimension::CubeArray, false) => {
                                GlTextureTarget::CubeArray
                            }
                            _ => {
                                return Err(crate::backend::api::GlError::Unsupported {
                                    operation: "bind reflected GL group",
                                    reason: "sampled texture dimension",
                                });
                            }
                        };
                        provider.active_texture(unit)?;
                        provider.bind_texture(unit, target, Some(view.binding_texture().id))?;
                        if matches!(
                            texture_desc.format,
                            crate::backend::api::GlFormat::Depth24PlusStencil8
                                | crate::backend::api::GlFormat::Depth32FloatStencil8
                        ) {
                            provider.set_depth_stencil_texture_mode(
                                view.binding_texture().id,
                                view.aspects == fluxel_rhi_core::api::TextureAspects::STENCIL,
                            )?;
                            // Selecting the depth/stencil plane is also used
                            // by cold view creation, where the provider
                            // canonical-cleans its temporary texture binding.
                            // Reinstall the sampled binding after that helper
                            // invalidates its CPU mirror; otherwise stencil
                            // mirror replay leaves this unit bound to zero.
                            provider.active_texture(unit)?;
                            provider.bind_texture(unit, target, Some(view.binding_texture().id))?;
                        }
                        let has_sampler = reflection.assignments.iter().any(|other| {
                            other.executable == assignment.executable
                                && layout.native.bindings.iter().any(|logical| {
                                    logical.location == other.logical
                                        && matches!(
                                            logical.kind,
                                            crate::backend::api::GlShaderResourceKind::Sampler
                                        )
                                })
                        });
                        if !has_sampler {
                            provider.bind_sampler(unit, None)?;
                        }
                    }
                    (
                        GlBindingSlot::StorageTexture {
                            access,
                            dimension,
                            format,
                        },
                        GlExecutableBindingLocation::ImageUnit(unit),
                    ) => {
                        let access = match access {
                            fluxel_rhi_core::api::StorageAccess::Read => {
                                crate::backend::api::GlStorageImageAccess::ReadOnly
                            }
                            fluxel_rhi_core::api::StorageAccess::Write => {
                                crate::backend::api::GlStorageImageAccess::WriteOnly
                            }
                            fluxel_rhi_core::api::StorageAccess::ReadWrite => {
                                crate::backend::api::GlStorageImageAccess::ReadWrite
                            }
                        };
                        let resource = reflected_resource(group, binding, assignment.element);
                        if resource.is_none()
                            && allows_missing_element(entry)
                            && assignment.element.is_some()
                        {
                            let dimension = match dimension {
                                fluxel_rhi_core::api::TextureViewDimension::D2 => {
                                    crate::backend::api::GlTextureDimension::D2
                                }
                                fluxel_rhi_core::api::TextureViewDimension::D2Array => {
                                    crate::backend::api::GlTextureDimension::D2Array
                                }
                                fluxel_rhi_core::api::TextureViewDimension::D3 => {
                                    crate::backend::api::GlTextureDimension::D3
                                }
                                _ => {
                                    return Err(crate::backend::api::GlError::Unsupported {
                                        operation: "bind reflected GL group",
                                        reason: "dummy storage image dimension",
                                    });
                                }
                            };
                            let format = crate::backend::translate::lower_texture_format(format)
                                .map_err(|_| crate::backend::api::GlError::Unsupported {
                                    operation: "bind reflected GL group",
                                    reason: "dummy storage image format",
                                })?;
                            provider.bind_dummy_storage_image(unit, dimension, format, access)?;
                            continue;
                        }
                        let Some((_, GlBoundResource::StorageTexture(view))) = resource else {
                            return Err(crate::backend::api::GlError::Driver {
                                operation: "bind reflected GL group",
                                message: "storage texture is absent".into(),
                            });
                        };
                        provider.bind_storage_image(
                            unit,
                            crate::backend::api::GlStorageImageBinding {
                                texture: view.binding_texture().id,
                                level: view.native.mip_level,
                                sample_count: view.native.sample_count,
                                layered: matches!(
                                    dimension,
                                    fluxel_rhi_core::api::TextureViewDimension::D2Array
                                        | fluxel_rhi_core::api::TextureViewDimension::D3
                                ),
                                layer: if matches!(
                                    dimension,
                                    fluxel_rhi_core::api::TextureViewDimension::D2Array
                                        | fluxel_rhi_core::api::TextureViewDimension::D3
                                ) {
                                    None
                                } else {
                                    Some(view.native.array_layer)
                                },
                                format: view.native.format,
                                access,
                            },
                        )?;
                    }
                    (
                        GlBindingSlot::Sampler { comparison, .. },
                        GlExecutableBindingLocation::TextureUnit(unit),
                    ) => {
                        let resource = reflected_resource(group, binding, assignment.element);
                        if resource.is_none()
                            && allows_missing_element(entry)
                            && assignment.element.is_some()
                        {
                            provider.bind_dummy_sampler(unit, comparison)?;
                            continue;
                        }
                        let Some((_, GlBoundResource::Sampler(sampler))) = resource else {
                            return Err(crate::backend::api::GlError::Driver {
                                operation: "bind reflected GL group",
                                message: "sampler is absent".into(),
                            });
                        };
                        provider.bind_sampler(unit, Some(sampler.id))?;
                    }
                    _ => {
                        return Err(crate::backend::api::GlError::Unsupported {
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
        remember_group(encoder, index, group, offsets);
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
    use crate::backend::api::GlExecutableBindingLocation;
    let browser = encoder.browser.as_mut().ok_or(Error::InvalidState)?;
    let result = (|| -> Result<()> {
        for uniform in pipeline
            .mip_uniforms
            .iter()
            .filter(|uniform| uniform.group == index)
        {
            let count = mip_count_for_binding(group, uniform)?;
            browser.set_texture_mip_count(&pipeline.browser_program, &uniform.name, count)?;
        }
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
            match (entry.slot, assignment.executable) {
                (
                    GlBindingSlot::UniformBuffer { .. },
                    GlExecutableBindingLocation::UniformBlock(point),
                ) => {
                    if reflected_resource(group, binding, assignment.element).is_none()
                        && allows_missing_element(entry)
                        && assignment.element.is_some()
                    {
                        browser.bind_dummy_uniform_buffer(&pipeline.browser_program, point)?;
                        continue;
                    }
                    let Some((
                        _,
                        GlBoundResource::UniformBuffer {
                            buffer,
                            offset,
                            size,
                        },
                    )) = reflected_resource(group, binding, assignment.element)
                    else {
                        return Err(Error::InvalidState);
                    };
                    let offset = offset
                        .checked_add(u64::from(dynamic_offset(group, binding, offsets)?))
                        .ok_or(Error::InvalidRange)?;
                    let alignment = encoder
                        .state
                        .owner
                        .discovery_snapshot()
                        .limits()
                        .uniform_buffer_offset_alignment;
                    if alignment == 0 || offset % alignment != 0 {
                        return Err(Error::InvalidRange);
                    }
                    browser.bind_uniform_buffer_named(
                        &pipeline.browser_program,
                        point,
                        &buffer.browser,
                        offset,
                        *size,
                    )?;
                }
                (
                    GlBindingSlot::SampledTexture {
                        dimension,
                        sample,
                        multisampled,
                    },
                    GlExecutableBindingLocation::TextureUnit(unit),
                ) => {
                    match reflected_resource(group, binding, assignment.element) {
                        Some((_, GlBoundResource::Texture(view))) => {
                            let texture = view
                                .stencil_mirror
                                .as_ref()
                                .or(view.cube_mirror.as_ref())
                                .unwrap_or(&view.texture.browser);
                            browser.bind_sampled_texture_only(unit, texture)?;
                            if matches!(
                                view.binding_texture().desc.format,
                                crate::backend::api::GlFormat::Depth24PlusStencil8
                                    | crate::backend::api::GlFormat::Depth32FloatStencil8
                            ) {
                                browser.set_depth_stencil_texture_mode(
                                    texture,
                                    view.aspects == fluxel_rhi_core::api::TextureAspects::STENCIL,
                                )?;
                            }
                        }
                        None if allows_missing_element(entry) && assignment.element.is_some() => {
                            browser.bind_dummy_sampled_texture(
                                unit,
                                dimension,
                                sample,
                                multisampled,
                            )?
                        }
                        _ => return Err(Error::InvalidState),
                    }
                    let has_sampler = pipeline.reflection.assignments.iter().any(|other| {
                        other.executable == assignment.executable
                            && pipeline.layout.native.bindings.iter().any(|logical| {
                                logical.location == other.logical
                                    && matches!(
                                        logical.kind,
                                        crate::backend::api::GlShaderResourceKind::Sampler
                                    )
                            })
                    });
                    if !has_sampler {
                        browser.bind_sampler_only(unit, None)?;
                    }
                }
                (
                    GlBindingSlot::Sampler { comparison, .. },
                    GlExecutableBindingLocation::TextureUnit(unit),
                ) => match reflected_resource(group, binding, assignment.element) {
                    Some((_, GlBoundResource::Sampler(sampler))) => {
                        browser.bind_sampler_only(unit, Some(&sampler.browser))?
                    }
                    None if allows_missing_element(entry) && assignment.element.is_some() => {
                        browser.bind_dummy_sampler(unit, comparison)?
                    }
                    _ => return Err(Error::InvalidState),
                },
                _ => return Err(Error::Unsupported("WebGL2 reflected binding kind")),
            }
        }
        Ok(())
    })();
    if result.is_ok() {
        remember_group(encoder, index, group, offsets);
        encoder.issued = true;
    } else {
        encoder.indeterminate = true;
    }
    result
}
