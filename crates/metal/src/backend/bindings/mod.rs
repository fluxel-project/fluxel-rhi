//! Metal bind-group resources and pipeline-layout metadata.
//!
//! WGSL and native Metal shaders use the same pipeline-specific stage maps.
//! Resource ownership and encoder binding are implemented by their respective
//! native paths.

use fluxel_rhi_core::api::{
    self, BindGroupLayoutDesc, BindGroupLayoutEntry, BindingType, PipelineLayoutDesc, ShaderStage,
};

#[cfg(any(target_vendor = "apple", test))]
pub(crate) const UNIFORM_OFFSET_ALIGNMENT: u64 = 32;

#[cfg(target_vendor = "apple")]
pub(crate) fn uniform_offset_alignment(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> u64 {
    use objc2_metal::{MTLDevice as _, MTLGPUFamily};
    if device.supportsFamily(MTLGPUFamily::Apple2) {
        4
    } else {
        32
    }
}

#[cfg(target_vendor = "apple")]
pub(crate) fn argument_arrays_supported(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
) -> bool {
    use objc2_metal::{MTLArgumentBuffersTier, MTLDevice as _};
    device.argumentBuffersSupport() == MTLArgumentBuffersTier::Tier2
}

/// Enforce Metal's per-stage argument-buffer resource limits over the whole layout.
#[cfg(target_vendor = "apple")]
pub(crate) fn validate_array_limits(
    device: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>,
    layout: &MetalPipelineLayout,
    stage: ShaderStage,
) -> api::Result<()> {
    use objc2_metal::{MTLDevice as _, MTLGPUFamily as F};
    let apple9 = device.supportsFamily(F::Apple9);
    let apple7 = device.supportsFamily(F::Apple7);
    let apple6 = device.supportsFamily(F::Apple6);
    let apple4 = device.supportsFamily(F::Apple4);
    let mac = device.supportsFamily(F::Mac2);
    let buffer_limit = if apple6 || mac {
        u64::MAX
    } else if apple4 {
        96
    } else {
        31
    };
    let texture_limit = if apple6 || mac {
        1_000_000
    } else if apple4 {
        96
    } else {
        31
    };
    let sampler_limit = if apple9 {
        500_000
    } else if mac {
        1024
    } else if apple7 {
        996
    } else if apple6 {
        128
    } else {
        16
    };
    let mut counts = [0u64; 3];
    for entry in layout
        .groups
        .iter()
        .flat_map(|group| &group.entries)
        .filter(|entry| entry.visibility.contains(&stage))
    {
        let count = match entry.count {
            api::BindingCount::One => continue,
            api::BindingCount::Fixed(n) => n,
            api::BindingCount::RuntimeSized { max_count } => max_count,
        };
        if !argument_arrays_supported(device) {
            return Err(api::Error::Unsupported("Metal argument buffer tier 2"));
        }
        let index = match entry.ty {
            BindingType::UniformBuffer { .. }
            | BindingType::StorageBuffer { .. }
            | BindingType::AccelerationStructure => 0,
            BindingType::SampledTexture { .. } | BindingType::StorageTexture { .. } => 1,
            BindingType::Sampler { .. } => 2,
            BindingType::VideoFrame => return Err(api::Error::Unsupported("video frame binding")),
        };
        counts[index] = counts[index]
            .checked_add(u64::from(count))
            .ok_or(api::Error::InvalidRange)?;
    }
    if counts[0] > buffer_limit || counts[1] > texture_limit || counts[2] > sampler_limit {
        return Err(api::Error::Unsupported(
            "Metal per-stage argument buffer resource limit",
        ));
    }
    Ok(())
}

#[derive(Clone)]
pub struct MetalBindGroupLayout {
    #[allow(dead_code)] // Consumed when pipeline creation validates the fixed resource ABI.
    pub(crate) entries: Vec<BindGroupLayoutEntry>,
}

#[derive(Clone)]
pub struct MetalPipelineLayout {
    #[allow(dead_code)] // Retained for native pipeline and command binding validation.
    pub(crate) groups: Vec<MetalBindGroupLayout>,
    pub(crate) push_constants: Vec<api::PushConstantRange>,
}

#[derive(Clone)]
pub struct MetalBindGroup {
    #[cfg(target_vendor = "apple")]
    #[allow(dead_code)] // Read by the direct Metal render/compute encoder lowering.
    pub(crate) resources: Vec<BoundResource>,
}

#[cfg(target_vendor = "apple")]
#[derive(Clone)]
#[allow(dead_code)] // Resource handles are consumed by the next encoder implementation slice.
pub(crate) enum BoundResource {
    Array {
        binding: u32,
        visibility: Vec<api::ShaderStage>,
        ty: api::BindingType,
        elements: Vec<Option<BoundResource>>,
    },
    Acceleration {
        binding: u32,
        visibility: Vec<api::ShaderStage>,
        value: super::ray::MetalAcceleration,
    },
    Buffer {
        binding: u32,
        visibility: Vec<api::ShaderStage>,
        raw: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        offset: u64,
        size: u64,
        allocation_size: u64,
        dynamic_offset: bool,
        offset_alignment: u64,
        view: Option<api::StorageBufferView>,
    },
    Texture {
        binding: u32,
        visibility: Vec<api::ShaderStage>,
        raw: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLTexture>>,
    },
    Sampler {
        binding: u32,
        visibility: Vec<api::ShaderStage>,
        raw: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSamplerState>>,
    },
}

/// Convert one validated binding array into the per-element Metal wrapper ABI.
#[cfg(target_vendor = "apple")]
pub(crate) fn argument_elements(
    ty: &BindingType,
    elements: &[Option<BoundResource>],
    capacity: u32,
    dynamic_offset: Option<u32>,
) -> api::Result<Vec<super::argument::ArgumentElement>> {
    use super::argument::ArgumentElement as A;
    let count = usize::try_from(capacity).map_err(|_| api::Error::InvalidRange)?;
    if count == 0 {
        return Err(api::Error::InvalidRange);
    }
    let empty = || match ty {
        BindingType::UniformBuffer { .. } | BindingType::StorageBuffer { .. } => Ok(A::Buffer {
            raw: None,
            offset: 0,
        }),
        BindingType::SampledTexture { .. } | BindingType::StorageTexture { .. } => {
            Ok(A::Texture(None))
        }
        BindingType::Sampler { .. } => Ok(A::Sampler(None)),
        BindingType::AccelerationStructure => Ok(A::Acceleration(None)),
        BindingType::VideoFrame => Err(api::Error::Unsupported("video frame binding")),
    };
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| api::Error::OutOfMemory)?;
    for element in elements.iter().take(count) {
        result.push(match element {
            None => empty()?,
            Some(BoundResource::Buffer {
                raw,
                offset,
                size,
                allocation_size,
                dynamic_offset: dynamic,
                offset_alignment,
                ..
            }) => {
                let start = if *dynamic {
                    apply_dynamic_offset(
                        *offset,
                        *size,
                        *allocation_size,
                        dynamic_offset.ok_or(api::Error::InvalidRange)?,
                        *offset_alignment,
                    )?
                } else {
                    *offset
                };
                A::Buffer {
                    raw: Some(raw.clone()),
                    offset: usize::try_from(start).map_err(|_| api::Error::InvalidRange)?,
                }
            }
            Some(BoundResource::Texture { raw, .. }) => A::Texture(Some(raw.clone())),
            Some(BoundResource::Sampler { raw, .. }) => A::Sampler(Some(raw.clone())),
            Some(BoundResource::Acceleration { value, .. }) => {
                A::Acceleration(Some(value.raw.clone()))
            }
            Some(BoundResource::Array { .. }) => return Err(api::Error::InvalidState),
        });
    }
    while result.len() < count {
        result.push(empty()?);
    }
    Ok(result)
}

pub(super) fn create_bind_group(
    desc: &api::BindGroupDesc<'_, super::Metal>,
) -> api::Result<MetalBindGroup> {
    #[cfg(target_vendor = "apple")]
    {
        let layout =
            // SAFETY: Core checks layout and every resource belong to the same device.
            unsafe { desc.layout.native_ref() };
        let mut resources = Vec::with_capacity(desc.entries.len());
        // Core requires strictly increasing layout bindings and validates each
        // group entry against that order, which preserves dynamic-offset order.
        for (layout_entry, entry) in layout.entries.iter().zip(desc.entries) {
            let resource = match &entry.resource {
                api::BindingResource::Array(values) => BoundResource::Array {
                    binding: entry.binding,
                    visibility: layout_entry.visibility.clone(),
                    ty: layout_entry.ty.clone(),
                    elements: values
                        .iter()
                        .map(|value| {
                            value
                                .as_ref()
                                .map(|v| bind_one(layout_entry, entry.binding, v))
                                .transpose()
                        })
                        .collect::<api::Result<_>>()?,
                },
                value => bind_one(layout_entry, entry.binding, value)?,
            };
            resources.push(resource);
        }
        Ok(MetalBindGroup { resources })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        let _ = desc;
        Err(api::Error::Unsupported("Metal requires an Apple target"))
    }
}

#[cfg(target_vendor = "apple")]
fn bind_one(
    layout_entry: &BindGroupLayoutEntry,
    binding: u32,
    resource: &api::BindingResource<'_, super::Metal>,
) -> api::Result<BoundResource> {
    let resource = match resource {
        api::BindingResource::Buffer { buffer, range } => {
            // SAFETY: Core checked the buffer's device identity before
            // invoking this backend SPI; the native wrapper is cloned below.
            let native = unsafe { buffer.native_ref() };
            let (minimum, dynamic_offset) = match layout_entry.ty {
                BindingType::UniformBuffer {
                    min_size,
                    dynamic_offset,
                } => (min_size, dynamic_offset),
                BindingType::StorageBuffer {
                    min_size,
                    dynamic_offset,
                    ..
                } => {
                    if !valid_storage_buffer_binding(
                        native.usage,
                        range.start,
                        range.end,
                        native.size(),
                        min_size,
                    ) {
                        return Err(api::Error::InvalidRange);
                    }
                    return Ok(BoundResource::Buffer {
                        binding: binding,
                        visibility: layout_entry.visibility.clone(),
                        raw: native.raw.clone(),
                        offset: range.start,
                        size: range.end - range.start,
                        allocation_size: native.size(),
                        dynamic_offset,
                        offset_alignment: 4,
                        view: None,
                    });
                }
                _ => return Err(api::Error::InvalidState),
            };
            if !valid_uniform_buffer_binding(
                native.usage,
                range.start,
                range.end,
                native.size(),
                minimum,
                uniform_offset_alignment(&native.device._device),
            ) {
                return Err(api::Error::InvalidRange);
            }
            BoundResource::Buffer {
                binding: binding,
                visibility: layout_entry.visibility.clone(),
                raw: native.raw.clone(),
                offset: range.start,
                size: range.end - range.start,
                allocation_size: native.size(),
                dynamic_offset,
                offset_alignment: uniform_offset_alignment(&native.device._device),
                view: None,
            }
        }
        api::BindingResource::TextureView(view) => {
            // SAFETY: Core checked the view's device identity before this SPI call.
            let native = unsafe { view.native_ref() };
            let (dimension, storage) = match layout_entry.ty {
                BindingType::SampledTexture {
                    dimension,
                    multisampled,
                    sample,
                } => {
                    let type_matches = if native.aspects == api::TextureAspects::STENCIL {
                        sample == api::TextureSampleType::Uint
                    } else {
                        sample_type_matches(sample, native.format)
                    };
                    if multisampled != (native.samples > 1)
                        || !type_matches
                        || (sample == api::TextureSampleType::Float && !native.filterable)
                    {
                        return Err(api::Error::InvalidState);
                    }
                    (dimension, false)
                }
                BindingType::StorageTexture {
                    format, dimension, ..
                } if format == native.format && native.samples == 1 => (dimension, true),
                _ => return Err(api::Error::InvalidState),
            };
            let usage = if storage {
                api::TextureUsage::STORAGE_BINDING
            } else {
                api::TextureUsage::TEXTURE_BINDING
            };
            if !native.usage.contains(usage) || native.dimension != dimension {
                return Err(api::Error::InvalidState);
            }
            BoundResource::Texture {
                binding: binding,
                visibility: layout_entry.visibility.clone(),
                raw: native.sampled_raw.clone(),
            }
        }
        api::BindingResource::Sampler(sampler) => {
            // SAFETY: Core checked the sampler's device identity before this SPI call.
            let native = unsafe { sampler.native_ref() };
            let (filtering, comparison) = match layout_entry.ty {
                BindingType::Sampler {
                    filtering,
                    comparison,
                } => (filtering, comparison),
                _ => return Err(api::Error::InvalidState),
            };
            if native.filtering != filtering || native.comparison != comparison {
                return Err(api::Error::InvalidState);
            }
            BoundResource::Sampler {
                binding: binding,
                visibility: layout_entry.visibility.clone(),
                raw: native.raw.clone(),
            }
        }
        api::BindingResource::Acceleration(accel) => {
            if !matches!(layout_entry.ty, BindingType::AccelerationStructure) {
                return Err(api::Error::InvalidState);
            }
            // SAFETY: Core validated device identity before this SPI call.
            let value = unsafe { accel.native_ref().clone() };
            BoundResource::Acceleration {
                binding: binding,
                visibility: layout_entry.visibility.clone(),
                value,
            }
        }
        api::BindingResource::StorageBufferView {
            buffer,
            range,
            view,
        } => {
            let (minimum, dynamic_offset) = match layout_entry.ty {
                BindingType::StorageBuffer {
                    min_size,
                    dynamic_offset,
                    ..
                } => (min_size, dynamic_offset),
                _ => return Err(api::Error::InvalidState),
            };
            // SAFETY: Core validated this buffer's device identity.
            let native = unsafe { buffer.native_ref() };
            if !valid_storage_buffer_binding(
                native.usage,
                range.start,
                range.end,
                native.size(),
                minimum,
            ) {
                return Err(api::Error::InvalidRange);
            }
            BoundResource::Buffer {
                binding: binding,
                visibility: layout_entry.visibility.clone(),
                raw: native.raw.clone(),
                offset: range.start,
                size: range.end - range.start,
                allocation_size: native.size(),
                dynamic_offset,
                offset_alignment: 4,
                view: Some(*view),
            }
        }
        api::BindingResource::Array(_) => return Err(api::Error::InvalidState),
    };
    Ok(resource)
}

pub(super) fn create_layout(desc: &BindGroupLayoutDesc) -> api::Result<MetalBindGroupLayout> {
    for entry in &desc.entries {
        if matches!(entry.ty, BindingType::AccelerationStructure)
            && entry.visibility.iter().any(|stage| {
                !matches!(
                    stage,
                    ShaderStage::Compute | ShaderStage::Vertex | ShaderStage::Fragment
                )
            })
        {
            return Err(api::Error::Unsupported("Metal acceleration binding stage"));
        }
        if entry.visibility.is_empty()
            || entry.visibility.iter().any(|stage| {
                !matches!(
                    stage,
                    ShaderStage::Vertex
                        | ShaderStage::Fragment
                        | ShaderStage::Compute
                        | ShaderStage::Task
                        | ShaderStage::Mesh
                )
            })
            || !supported_type(&entry.ty)
        {
            return Err(api::Error::Unsupported("Metal bind group layout entry"));
        }
    }
    Ok(MetalBindGroupLayout {
        entries: desc.entries.clone(),
    })
}

pub(super) fn create_pipeline_layout(
    desc: &PipelineLayoutDesc<'_, super::Metal>,
) -> api::Result<MetalPipelineLayout> {
    Ok(MetalPipelineLayout {
        groups: desc
            .groups
            .iter()
            .map(|group| {
                // SAFETY: Core validated every group's DeviceId before calling
                // this SPI; the borrowed native metadata is cloned immediately.
                unsafe { group.native_ref().clone() }
            })
            .collect(),
        push_constants: desc.push_constants.to_vec(),
    })
}

fn supported_type(ty: &BindingType) -> bool {
    match ty {
        BindingType::VideoFrame => false,
        _ => true,
    }
}

// The shader ABI supplies the selected range length to Naga's bounds checks.
#[allow(dead_code)]
fn valid_storage_buffer_binding(
    usage: api::BufferUsage,
    start: u64,
    end: u64,
    allocation_size: u64,
    minimum_size: u64,
) -> bool {
    usage.contains(api::BufferUsage::STORAGE)
        && start.is_multiple_of(4)
        && start < end
        && end <= allocation_size
        && end - start >= minimum_size
}

#[allow(dead_code)] // Used by the Apple-only bind-group materialization path.
fn sample_type_matches(sample: api::TextureSampleType, format: api::TextureFormat) -> bool {
    use api::{TextureFormat as F, TextureSampleType as S};
    match sample {
        S::Depth => matches!(
            format,
            F::Depth16Unorm
                | F::Depth24Plus
                | F::Depth24PlusStencil8
                | F::Depth32Float
                | F::Depth32FloatStencil8
        ),
        S::Sint => matches!(
            format,
            F::R8Sint
                | F::Rg8Sint
                | F::Rgba8Sint
                | F::R16Sint
                | F::Rg16Sint
                | F::Rgba16Sint
                | F::R32Sint
                | F::Rg32Sint
                | F::Rgba32Sint
        ),
        S::Uint => matches!(
            format,
            F::R8Uint
                | F::Rg8Uint
                | F::Rgba8Uint
                | F::R16Uint
                | F::Rg16Uint
                | F::Rgba16Uint
                | F::R32Uint
                | F::Rg32Uint
                | F::Rgba32Uint
                | F::Rgb10a2Uint
                | F::R64Uint
                | F::Rg64Uint
                | F::Stencil8
        ),
        S::Float | S::UnfilterableFloat => {
            !sample_type_matches(S::Depth, format)
                && !sample_type_matches(S::Sint, format)
                && !sample_type_matches(S::Uint, format)
        }
    }
}

#[allow(dead_code)] // Used by the Apple-only bind-group materialization path.
fn valid_uniform_buffer_binding(
    usage: api::BufferUsage,
    start: u64,
    end: u64,
    allocation_size: u64,
    minimum_size: u64,
    offset_alignment: u64,
) -> bool {
    start <= end
        && end <= allocation_size
        && offset_alignment != 0
        && start.is_multiple_of(offset_alignment)
        && usage.contains(api::BufferUsage::UNIFORM)
        && end - start >= minimum_size
}

/// Applies and validates a dynamic offset while preserving the bound range.
#[cfg(any(target_vendor = "apple", test))]
pub(crate) fn apply_dynamic_uniform_offset(
    bound_start: u64,
    bound_size: u64,
    allocation_size: u64,
    dynamic_offset: u32,
) -> api::Result<u64> {
    apply_dynamic_offset(
        bound_start,
        bound_size,
        allocation_size,
        dynamic_offset,
        UNIFORM_OFFSET_ALIGNMENT,
    )
}

#[cfg(any(target_vendor = "apple", test))]
pub(crate) fn apply_dynamic_offset(
    bound_start: u64,
    bound_size: u64,
    allocation_size: u64,
    dynamic_offset: u32,
    alignment: u64,
) -> api::Result<u64> {
    if alignment == 0 || !u64::from(dynamic_offset).is_multiple_of(alignment) {
        return Err(api::Error::InvalidRange);
    }
    let start = bound_start
        .checked_add(u64::from(dynamic_offset))
        .ok_or(api::Error::InvalidRange)?;
    if start
        .checked_add(bound_size)
        .is_none_or(|end| end > allocation_size)
    {
        return Err(api::Error::InvalidRange);
    }
    Ok(start)
}

#[cfg(test)]
#[path = "../../tests/backend/bindings.rs"]
mod tests;
