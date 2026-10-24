//! Synchronous WebGPU bind-group admission from retained native descriptors.

use fluxel_rhi_core::api::{self, Error, Result};

use super::{
    instance::WebGpuDevice,
    resource::{WebGpuBuffer, WebGpuSampler, WebGpuTextureView, device_limit},
};

pub(super) fn validate(
    device: &WebGpuDevice,
    layout: &api::BindGroupLayoutEntry,
    resource: &api::BindingResource<'_, super::instance::WebGpu>,
) -> Result<()> {
    match (&layout.ty, resource) {
        (
            api::BindingType::UniformBuffer {
                dynamic_offset: _,
                min_size,
            },
            api::BindingResource::Buffer { buffer, range },
        ) => {
            // SAFETY: Core checked the resource device before this backend call.
            buffer_binding(
                device,
                unsafe { buffer.native_ref() },
                range,
                *min_size,
                true,
            )
        }
        (
            api::BindingType::StorageBuffer {
                dynamic_offset: _,
                min_size,
                ..
            },
            api::BindingResource::Buffer { buffer, range }
            | api::BindingResource::StorageBufferView { buffer, range, .. },
        ) => buffer_binding(
            device,
            // SAFETY: Core checked the resource device before this backend call.
            unsafe { buffer.native_ref() },
            range,
            *min_size,
            false,
        ),
        (
            api::BindingType::SampledTexture {
                sample,
                dimension,
                multisampled,
            },
            api::BindingResource::TextureView(view),
        ) => {
            // SAFETY: Core checked the view device before this backend call.
            sampled_texture(
                device,
                unsafe { view.native_ref() },
                *sample,
                *dimension,
                *multisampled,
            )
        }
        (
            api::BindingType::StorageTexture {
                format, dimension, ..
            },
            api::BindingResource::TextureView(view),
        ) => {
            // SAFETY: Core checked the view device before this backend call.
            storage_texture(unsafe { view.native_ref() }, *format, *dimension)
        }
        (
            api::BindingType::Sampler {
                filtering,
                comparison,
            },
            api::BindingResource::Sampler(sampler),
        ) => {
            // SAFETY: Core checked the sampler device before this backend call.
            validate_sampler(unsafe { sampler.native_ref() }, *filtering, *comparison)
        }
        _ => Err(Error::InvalidState),
    }
}

fn buffer_binding(
    device: &WebGpuDevice,
    buffer: &WebGpuBuffer,
    range: &std::ops::Range<u64>,
    min_size: u64,
    uniform: bool,
) -> Result<()> {
    let native_usage = if uniform { 64 } else { 128 };
    let max = device_limit(
        &device.0.device,
        if uniform {
            "maxUniformBufferBindingSize"
        } else {
            "maxStorageBufferBindingSize"
        },
    );
    let alignment = device_limit(
        &device.0.device,
        if uniform {
            "minUniformBufferOffsetAlignment"
        } else {
            "minStorageBufferOffsetAlignment"
        },
    );
    if buffer.native_usage & native_usage == 0
        || range.start >= range.end
        || range.end > buffer.size
        || range.end - range.start < min_size
        || range.end - range.start > max
        || (!uniform && (range.end - range.start) % 4 != 0)
        || (alignment != 0 && range.start % alignment != 0)
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

fn sampled_texture(
    device: &WebGpuDevice,
    view: &WebGpuTextureView,
    sample: api::TextureSampleType,
    dimension: api::TextureViewDimension,
    multisampled: bool,
) -> Result<()> {
    let texture = &view._texture;
    if !texture
        .desc
        .usage
        .contains(api::TextureUsage::TEXTURE_BINDING)
        || view.desc.dimension != dimension
        || (texture.desc.samples > 1) != multisampled
        || !sample_matches(device, view, sample)
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}

fn storage_texture(
    view: &WebGpuTextureView,
    format: api::TextureFormat,
    dimension: api::TextureViewDimension,
) -> Result<()> {
    if !view
        ._texture
        .desc
        .usage
        .contains(api::TextureUsage::STORAGE_BINDING)
        || view.desc.dimension != dimension
        || view.desc.format != format
        || view._texture.desc.samples != 1
    {
        return Err(Error::InvalidState);
    }
    Ok(())
}

fn validate_sampler(sampler: &WebGpuSampler, filtering: bool, comparison: bool) -> Result<()> {
    let actual_comparison = sampler.desc.compare.is_some();
    let actual_filtering = matches!(sampler.desc.mag_filter, api::FilterMode::Linear)
        || matches!(sampler.desc.min_filter, api::FilterMode::Linear)
        || matches!(sampler.desc.mip_filter, api::FilterMode::Linear);
    // A filtering layout accepts both filtering and nearest samplers; only a
    // non-filtering layout must reject an actual filtering sampler.
    if comparison != actual_comparison || (!comparison && !filtering && actual_filtering) {
        return Err(Error::InvalidState);
    }
    Ok(())
}

fn sample_matches(
    device: &WebGpuDevice,
    view: &WebGpuTextureView,
    sample: api::TextureSampleType,
) -> bool {
    let format = view.desc.format;
    let aspects = view.desc.aspects;
    let depth = api::TextureAspects::DEPTH;
    let stencil = api::TextureAspects::STENCIL;
    // WebGPU sampled bindings have one sample class. A combined aspect view
    // cannot describe either the depth or stencil class unambiguously.
    if aspects == (depth | stencil) {
        return false;
    }
    if aspects == stencil {
        return matches!(sample, api::TextureSampleType::Uint);
    }
    if aspects == depth {
        return matches!(sample, api::TextureSampleType::Depth);
    }
    let name = super::resource::texture_format(format).unwrap_or("");
    match sample {
        api::TextureSampleType::Sint => name.ends_with("sint"),
        api::TextureSampleType::Uint => name.ends_with("uint"),
        api::TextureSampleType::Depth => name.starts_with("depth"),
        api::TextureSampleType::Float => {
            !name.ends_with("sint")
                && !name.ends_with("uint")
                && !name.starts_with("depth")
                && super::resource_validation::capabilities(device, format).0
        }
        api::TextureSampleType::UnfilterableFloat => {
            !name.ends_with("sint") && !name.ends_with("uint") && !name.starts_with("depth")
        }
    }
}
