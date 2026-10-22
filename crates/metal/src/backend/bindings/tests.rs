use super::{
    UNIFORM_OFFSET_ALIGNMENT, apply_dynamic_uniform_offset, create_layout, sample_type_matches,
    supported_type, valid_storage_buffer_binding, valid_uniform_buffer_binding,
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
fn layout_accepts_dynamic_storage_and_sampled_textures() {
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
    assert!(create_layout(&desc).is_ok());
    desc.entries[0] = entry(
        0,
        BindingType::SampledTexture {
            sample: TextureSampleType::Float,
            dimension: TextureViewDimension::D3,
            multisampled: false,
        },
    );
    assert!(supported_type(&desc.entries[0].ty));
}

#[test]
fn storage_buffers_support_raster_stages_and_subranges() {
    let mut item = entry(
        0,
        BindingType::StorageBuffer {
            dynamic_offset: false,
            min_size: 4,
            access: StorageAccess::ReadWrite,
        },
    );
    item.visibility = vec![ShaderStage::Compute];
    assert!(
        create_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![item.clone()]
        })
        .is_ok()
    );
    item.visibility.push(ShaderStage::Fragment);
    assert!(
        create_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![item]
        })
        .is_ok()
    );
    assert!(valid_storage_buffer_binding(
        api::BufferUsage::STORAGE,
        0,
        16,
        16,
        4
    ));
    assert!(valid_storage_buffer_binding(
        api::BufferUsage::STORAGE,
        4,
        16,
        16,
        4
    ));
    assert!(!valid_storage_buffer_binding(
        api::BufferUsage::COPY_DST,
        0,
        16,
        16,
        4
    ));
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
    assert!(supported_type(&unsupported_format.ty));
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
