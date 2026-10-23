//! Per-stage and combined native GL program resource limits.

use super::*;

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

pub(super) fn validate_combined_buffer_limits(
    device: &GlDevice,
    layout: &super::super::types::GlPipelineLayout,
    shaders: &[&super::super::types::GlShader],
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
                                    super::super::types::GlBindingSlot::StorageBuffer { .. }
                                )
                            } else {
                                matches!(
                                    entry.slot,
                                    super::super::types::GlBindingSlot::UniformBuffer { .. }
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

pub(super) fn validate_combined_sampled_limits(
    device: &GlDevice,
    shaders: &[&super::super::types::GlShader],
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
                            Some(super::super::types::GlNagaResourceType::SampledTexture { .. })
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

pub(super) fn validate_combined_image_limits(
    device: &GlDevice,
    shaders: &[&super::super::types::GlShader],
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
                        Some(super::super::types::GlNagaResourceType::StorageTexture { .. })
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
