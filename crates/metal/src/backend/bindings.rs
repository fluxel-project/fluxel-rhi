//! Metal's fixed bind-group ABI and pipeline-layout metadata.
//!
//! WGSL group `g`, binding `b` lowers to Metal resource slot `7 * g + b`.
//! Each group's binding numbers are kept unchanged so command recording can
//! bind each resource directly without building a CPU command list. This file
//! only accepts the scalar resource forms currently lowered by the shader ABI;
//! resource ownership and encoder binding are implemented by their own paths.

use fluxel_rhi_core::api::{
    self, BindGroupLayoutDesc, BindGroupLayoutEntry, BindingCount, BindingType, PipelineLayoutDesc,
    ShaderStage, TextureViewDimension,
};

const MAX_GROUPS: usize = 4;
const MAX_BINDINGS: u32 = 7;
#[cfg(any(target_vendor = "apple", test))]
pub(crate) const UNIFORM_OFFSET_ALIGNMENT: u64 = 32;

#[derive(Clone)]
pub struct MetalBindGroupLayout {
    #[allow(dead_code)] // Consumed when pipeline creation validates the fixed resource ABI.
    pub(crate) entries: Vec<BindGroupLayoutEntry>,
}

#[derive(Clone)]
pub struct MetalPipelineLayout {
    #[allow(dead_code)] // Retained for native pipeline and command binding validation.
    pub(crate) groups: Vec<MetalBindGroupLayout>,
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
    Buffer {
        binding: u32,
        visibility: Vec<api::ShaderStage>,
        raw: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        offset: u64,
        size: u64,
        allocation_size: u64,
        dynamic_offset: bool,
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
                api::BindingResource::Buffer { buffer, range } => {
                    // SAFETY: Core checked the buffer's device identity before
                    // invoking this backend SPI; the native wrapper is cloned below.
                    let native = unsafe { buffer.native_ref() };
                    let (minimum, dynamic_offset) = match layout_entry.ty {
                        BindingType::UniformBuffer {
                            min_size,
                            dynamic_offset,
                        } => (min_size, dynamic_offset),
                        _ => return Err(api::Error::InvalidState),
                    };
                    if !valid_uniform_buffer_binding(
                        native.usage,
                        range.start,
                        range.end,
                        native.size(),
                        minimum,
                        UNIFORM_OFFSET_ALIGNMENT,
                    ) {
                        return Err(api::Error::InvalidRange);
                    }
                    BoundResource::Buffer {
                        binding: entry.binding,
                        visibility: layout_entry.visibility.clone(),
                        raw: native.raw.clone(),
                        offset: range.start,
                        size: range.end - range.start,
                        allocation_size: native.size(),
                        dynamic_offset,
                    }
                }
                api::BindingResource::TextureView(view) => {
                    // SAFETY: Core checked the view's device identity before this SPI call.
                    let native = unsafe { view.native_ref() };
                    let (dimension, storage) = match layout_entry.ty {
                        BindingType::SampledTexture {
                            dimension,
                            multisampled,
                            ..
                        } if !multisampled => (dimension, false),
                        BindingType::StorageTexture {
                            format,
                            dimension: TextureViewDimension::D2,
                            access: api::StorageAccess::Write,
                        } if format == native.format => (TextureViewDimension::D2, true),
                        _ => return Err(api::Error::InvalidState),
                    };
                    let usage = if storage {
                        api::TextureUsage::STORAGE_BINDING
                    } else {
                        api::TextureUsage::TEXTURE_BINDING
                    };
                    if !native.usage.contains(usage)
                        || native.dimension != dimension
                        || (!storage
                            && !sample_type_matches(
                                match layout_entry.ty {
                                    BindingType::SampledTexture { sample, .. } => sample,
                                    _ => unreachable!(),
                                },
                                native.format,
                            ))
                    {
                        return Err(api::Error::InvalidState);
                    }
                    BoundResource::Texture {
                        binding: entry.binding,
                        visibility: layout_entry.visibility.clone(),
                        raw: native.raw.clone(),
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
                        binding: entry.binding,
                        visibility: layout_entry.visibility.clone(),
                        raw: native.raw.clone(),
                    }
                }
                api::BindingResource::Acceleration(_) | api::BindingResource::Array(_) => {
                    return Err(api::Error::Unsupported("Metal binding resource form"));
                }
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

pub(super) fn create_layout(desc: &BindGroupLayoutDesc) -> api::Result<MetalBindGroupLayout> {
    if desc.entries.len() > MAX_BINDINGS as usize {
        return Err(api::Error::Unsupported(
            "Metal bind group exceeds seven slots",
        ));
    }
    for entry in &desc.entries {
        if entry.binding >= MAX_BINDINGS
            || entry.count != BindingCount::One
            || entry.partially_bound
            || entry.nonuniform_indexing
            || entry.visibility.is_empty()
            || entry.visibility.iter().any(|stage| {
                !matches!(
                    stage,
                    ShaderStage::Vertex | ShaderStage::Fragment | ShaderStage::Compute
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
    if desc.groups.len() > MAX_GROUPS {
        return Err(api::Error::Unsupported(
            "Metal pipeline layout exceeds four groups",
        ));
    }
    if !desc.push_constants.is_empty() {
        return Err(api::Error::Unsupported("Metal push constants"));
    }
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
    })
}

fn supported_type(ty: &BindingType) -> bool {
    match ty {
        BindingType::UniformBuffer { .. } => true,
        BindingType::SampledTexture {
            dimension,
            multisampled,
            ..
        } => {
            !multisampled
                && matches!(
                    dimension,
                    TextureViewDimension::D2
                        | TextureViewDimension::D2Array
                        | TextureViewDimension::Cube
                        | TextureViewDimension::CubeArray
                )
        }
        BindingType::Sampler {
            filtering: true,
            comparison: false,
        } => true,
        BindingType::StorageTexture {
            format: api::TextureFormat::Rgba8Unorm,
            dimension: TextureViewDimension::D2,
            access: api::StorageAccess::Write,
        } => true,
        BindingType::Sampler { .. }
        | BindingType::StorageBuffer { .. }
        | BindingType::StorageTexture { .. }
        | BindingType::AccelerationStructure
        | BindingType::ExternalTexture => false,
    }
}

#[allow(dead_code)] // Used by the Apple-only bind-group materialization path.
fn sample_type_matches(sample: api::TextureSampleType, format: api::TextureFormat) -> bool {
    use api::{TextureFormat as F, TextureSampleType as S};
    match sample {
        S::Float | S::UnfilterableFloat => matches!(
            format,
            F::Rgba8Unorm | F::Rgba8UnormSrgb | F::Bgra8Unorm | F::Bgra8UnormSrgb | F::Rgba16Float
        ),
        S::Depth => format == F::Depth32Float,
        S::Sint | S::Uint => false,
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
    if !u64::from(dynamic_offset).is_multiple_of(UNIFORM_OFFSET_ALIGNMENT) {
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
mod tests {
    use super::{
        UNIFORM_OFFSET_ALIGNMENT, apply_dynamic_uniform_offset, create_layout, sample_type_matches,
        supported_type, valid_uniform_buffer_binding,
    };
    use fluxel_rhi_core::api::{
        self, BindGroupLayoutDesc, BindGroupLayoutEntry, BindingCount, BindingType, ShaderStage,
        StorageAccess, TextureSampleType, TextureViewDimension,
    };

    fn entry(binding: u32, ty: BindingType) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: vec![ShaderStage::Vertex, ShaderStage::Fragment],
            ty,
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        }
    }

    #[test]
    fn layout_accepts_dynamic_uniform_and_sampled_texture_slots() {
        let desc = BindGroupLayoutDesc {
            label: None,
            entries: vec![
                entry(
                    0,
                    BindingType::UniformBuffer {
                        dynamic_offset: true,
                        min_size: 16,
                    },
                ),
                entry(
                    6,
                    BindingType::SampledTexture {
                        sample: TextureSampleType::Float,
                        dimension: TextureViewDimension::D2,
                        multisampled: false,
                    },
                ),
            ],
        };
        assert!(create_layout(&desc).is_ok());
    }

    #[test]
    fn layout_rejects_forms_without_a_native_binding_lowering() {
        let mut desc = BindGroupLayoutDesc {
            label: None,
            entries: vec![entry(
                0,
                BindingType::StorageBuffer {
                    dynamic_offset: true,
                    min_size: 4,
                    access: StorageAccess::Read,
                },
            )],
        };
        assert!(matches!(
            create_layout(&desc),
            Err(api::Error::Unsupported(_))
        ));
        desc.entries[0] = entry(
            0,
            BindingType::SampledTexture {
                sample: TextureSampleType::Float,
                dimension: TextureViewDimension::D3,
                multisampled: false,
            },
        );
        assert!(!supported_type(&desc.entries[0].ty));
    }

    #[test]
    fn layout_accepts_compute_write_only_rgba8_storage_texture() {
        let mut storage = entry(
            1,
            BindingType::StorageTexture {
                format: api::TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                access: StorageAccess::Write,
            },
        );
        storage.visibility = vec![ShaderStage::Compute];
        let desc = BindGroupLayoutDesc {
            label: None,
            entries: vec![storage],
        };
        assert!(create_layout(&desc).is_ok());
        let unsupported_format = entry(
            0,
            BindingType::StorageTexture {
                format: api::TextureFormat::Rgba16Float,
                dimension: TextureViewDimension::D2,
                access: StorageAccess::Write,
            },
        );
        assert!(!supported_type(&unsupported_format.ty));
    }

    #[test]
    fn uniform_binding_requires_usage_valid_range_and_minimum_size() {
        assert!(valid_uniform_buffer_binding(
            api::BufferUsage::UNIFORM,
            256,
            320,
            512,
            64,
            UNIFORM_OFFSET_ALIGNMENT,
        ));
        assert!(!valid_uniform_buffer_binding(
            api::BufferUsage::COPY_SRC,
            0,
            64,
            64,
            16,
            UNIFORM_OFFSET_ALIGNMENT,
        ));
        assert!(!valid_uniform_buffer_binding(
            api::BufferUsage::UNIFORM,
            6,
            70,
            512,
            64,
            UNIFORM_OFFSET_ALIGNMENT,
        ));
        assert!(!valid_uniform_buffer_binding(
            api::BufferUsage::UNIFORM,
            4,
            68,
            512,
            64,
            UNIFORM_OFFSET_ALIGNMENT,
        ));
        assert!(!valid_uniform_buffer_binding(
            api::BufferUsage::UNIFORM,
            65,
            64,
            512,
            1,
            UNIFORM_OFFSET_ALIGNMENT,
        ));
        assert!(!valid_uniform_buffer_binding(
            api::BufferUsage::UNIFORM,
            0,
            15,
            512,
            16,
            UNIFORM_OFFSET_ALIGNMENT,
        ));
    }

    #[test]
    fn dynamic_uniform_offset_preserves_range_and_checks_alignment_and_bounds() {
        assert_eq!(apply_dynamic_uniform_offset(0, 64, 1024, 256), Ok(256));
        assert_eq!(apply_dynamic_uniform_offset(32, 64, 1024, 32), Ok(64));
        assert!(matches!(
            apply_dynamic_uniform_offset(0, 64, 1024, 4),
            Err(api::Error::InvalidRange)
        ));
        assert!(matches!(
            apply_dynamic_uniform_offset(0, 64, 1024, 2),
            Err(api::Error::InvalidRange)
        ));
        assert!(matches!(
            apply_dynamic_uniform_offset(960, 64, 1024, 4),
            Err(api::Error::InvalidRange)
        ));
        assert!(matches!(
            apply_dynamic_uniform_offset(u64::MAX - 3, 8, u64::MAX, 4),
            Err(api::Error::InvalidRange)
        ));
    }

    #[test]
    fn depth_and_color_sample_types_require_matching_native_formats() {
        assert!(sample_type_matches(
            api::TextureSampleType::Depth,
            api::TextureFormat::Depth32Float
        ));
        assert!(!sample_type_matches(
            api::TextureSampleType::Depth,
            api::TextureFormat::Rgba8Unorm
        ));
        assert!(!sample_type_matches(
            api::TextureSampleType::Uint,
            api::TextureFormat::Rgba8Unorm
        ));
    }
}
