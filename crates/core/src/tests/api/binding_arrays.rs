//! Core validation of runtime-sized binding array layouts and instances.
use super::{type_contract_tests::NativeShared, *};
use std::cell::Cell;

thread_local! {
    static LAYOUT_CALLS: Cell<usize> = const { Cell::new(0) };
    static GROUP_CALLS: Cell<usize> = const { Cell::new(0) };
    static TEXTURE_CALLS: Cell<usize> = const { Cell::new(0) };
}

impl ResourceBackend for NativeShared {
    fn buffer_support(_: &Arc<()>, _: &BufferDesc) -> Support<BufferSupport> {
        unreachable!()
    }
    fn texture_support(_: &Arc<()>, _: &TextureDesc) -> Support<TextureSupport> {
        Support::Supported(TextureSupport {
            max_extent: [4096, 4096, 1],
            max_array_layers: 1,
            max_mips: 1,
            sample_counts: vec![1, 4],
            compatible_view_formats: vec![TextureFormat::Rgba8Unorm],
            filterable: true,
            blendable: true,
            storage: false,
        })
    }
    fn create_buffer(_: &Arc<()>, _: &BufferDesc) -> Result<Arc<()>> {
        unreachable!()
    }
    fn create_texture(_: &Arc<()>, _: &TextureDesc) -> Result<Arc<()>> {
        TEXTURE_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(Arc::new(()))
    }
    fn create_view(_: &Arc<()>, _: &Arc<()>, _: &TextureViewDesc) -> Result<Arc<()>> {
        Ok(Arc::new(()))
    }
    fn create_sampler(_: &Arc<()>, _: &SamplerDesc) -> Result<Arc<()>> {
        Ok(Arc::new(()))
    }
    fn create_bind_group_layout(_: &Arc<()>, _: &BindGroupLayoutDesc) -> Result<Arc<()>> {
        LAYOUT_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(Arc::new(()))
    }
    fn create_bind_group(_: &Arc<()>, _: &BindGroupDesc<'_, Self>) -> Result<Arc<()>> {
        GROUP_CALLS.with(|calls| calls.set(calls.get() + 1));
        Ok(Arc::new(()))
    }
    fn create_pipeline_layout(_: &Arc<()>, _: &PipelineLayoutDesc<'_, Self>) -> Result<Arc<()>> {
        unreachable!()
    }
}

fn device(runtime_arrays: bool) -> Device<NativeShared> {
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.runtime_sized_binding_arrays = runtime_arrays;
    caps.nonuniform_sampler_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture supplies a native handle and capabilities for the same mock device.
    unsafe { Device::from_native(id, Arc::new(()), caps) }
}

fn entry(binding: u32, count: BindingCount, partially_bound: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: vec![],
        ty: BindingType::Sampler {
            filtering: false,
            comparison: false,
        },
        count,
        partially_bound,
        nonuniform_indexing: false,
    }
}

#[test]
fn transient_attachment_rejects_incompatible_creation_and_sampling_before_backend() {
    TEXTURE_CALLS.with(|calls| calls.set(0));
    GROUP_CALLS.with(|calls| calls.set(0));
    let device = device(true);
    let base = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [16, 16, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::TRANSIENT_ATTACHMENT | TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    let mut invalid = base.clone();
    invalid.usage |= TextureUsage::TEXTURE_BINDING;
    assert!(matches!(
        device.create_texture(&invalid),
        Err(Error::InvalidState)
    ));
    assert!(matches!(
        device.texture_support(&invalid),
        Support::Unsupported(_)
    ));
    TEXTURE_CALLS.with(|calls| assert_eq!(calls.get(), 0));

    let texture = device.create_texture(&base).unwrap();
    let view = device
        .create_view(
            &texture,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let layout = device
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Fragment],
                ty: BindingType::SampledTexture {
                    sample: TextureSampleType::Float,
                    dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    assert!(matches!(
        device.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&view),
            }],
        }),
        Err(Error::InvalidState)
    ));
    GROUP_CALLS.with(|calls| assert_eq!(calls.get(), 0));

    let lane = LaneId {
        device: device.id(),
        ordinal: 0,
    };
    let mut encoder = device.encoder(lane).unwrap();
    let invalid_load = [Some(ColorAttachment {
        view: AttachmentView::Texture(&view),
        resolve_target: None,
        load: LoadOp::Load,
        store: StoreOp::Discard,
    })];
    let raster = |colors| RasterPassDesc {
        label: None,
        colors,
        depth_stencil: None,
        render_area: Rect {
            x: 0,
            y: 0,
            width: 16,
            height: 16,
        },
        multiview_mask: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        secondary_inheritance: None,
    };
    assert!(matches!(
        encoder.begin_raster(&raster(&invalid_load)),
        Err(Error::InvalidState)
    ));

    let persistent_desc = TextureDesc {
        usage: TextureUsage::RENDER_ATTACHMENT,
        ..base.clone()
    };
    let persistent = device.create_texture(&persistent_desc).unwrap();
    let persistent_view = device
        .create_view(
            &persistent,
            &TextureViewDesc {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
        )
        .unwrap();
    let transient_resolve = [Some(ColorAttachment {
        view: AttachmentView::Texture(&persistent_view),
        resolve_target: Some(AttachmentView::Texture(&view)),
        load: LoadOp::Clear(ClearColor::Float([0.0; 4])),
        store: StoreOp::Store,
    })];
    assert!(matches!(
        encoder.begin_raster(&raster(&transient_resolve)),
        Err(Error::InvalidState)
    ));
}

#[test]
fn runtime_array_layout_validates_capacity_capability_and_position() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let device = device(true);
    let layout = |entries| BindGroupLayoutDesc {
        label: None,
        entries,
    };
    assert!(matches!(
        device.create_bind_group_layout(&layout(vec![entry(
            0,
            BindingCount::RuntimeSized { max_count: 0 },
            false
        )])),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        device.create_bind_group_layout(&layout(vec![
            entry(0, BindingCount::RuntimeSized { max_count: 2 }, false),
            entry(1, BindingCount::One, false)
        ])),
        Err(Error::InvalidState)
    ));
    assert!(matches!(
        self::device(false).create_bind_group_layout(&layout(vec![entry(
            0,
            BindingCount::RuntimeSized { max_count: 2 },
            false
        )])),
        Err(Error::Unsupported("runtime-sized binding arrays"))
    ));
    assert!(
        device
            .create_bind_group_layout(&layout(vec![
                entry(0, BindingCount::One, false),
                entry(1, BindingCount::RuntimeSized { max_count: 2 }, false)
            ]))
            .is_ok()
    );
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 1));
}

#[test]
fn storage_texture_nonuniform_indexing_uses_its_own_capability() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let layout = BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Compute],
            ty: BindingType::StorageTexture {
                format: TextureFormat::Rgba8Unorm,
                dimension: TextureViewDimension::D2,
                access: StorageAccess::Write,
            },
            count: BindingCount::Fixed(2),
            partially_bound: false,
            nonuniform_indexing: true,
        }],
    };
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    // The compatibility aggregate must not admit a storage texture by itself.
    caps.nonuniform_sampled_texture_and_storage_buffer_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture supplies matching capabilities for its mock device.
    let device_without_storage_texture_indexing =
        unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        device_without_storage_texture_indexing.create_bind_group_layout(&layout),
        Err(Error::Unsupported("nonuniform binding indexing"))
    ));

    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_storage_texture_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture supplies matching capabilities for its mock device.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(device.create_bind_group_layout(&layout).is_ok());
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 1));
}

#[test]
fn uniform_buffer_nonuniform_indexing_uses_its_own_capability() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let layout = BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Compute],
            ty: BindingType::UniformBuffer {
                dynamic_offset: false,
                min_size: 4,
            },
            count: BindingCount::Fixed(2),
            partially_bound: false,
            nonuniform_indexing: true,
        }],
    };
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_sampled_texture_and_storage_buffer_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // The compatibility aggregate must not admit a uniform buffer by itself.
    // SAFETY: this mock reports only the compatibility aggregate, not UBO indexing.
    let without = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        without.create_bind_group_layout(&layout),
        Err(Error::Unsupported("nonuniform binding indexing"))
    ));

    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_uniform_buffer_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this mock explicitly reports the UBO indexing capability.
    let with = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(with.create_bind_group_layout(&layout).is_ok());
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 1));
}

#[test]
fn storage_buffer_nonuniform_indexing_uses_its_own_capability() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let layout = BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Compute],
            ty: BindingType::StorageBuffer {
                dynamic_offset: false,
                min_size: 4,
                access: StorageAccess::Read,
            },
            count: BindingCount::Fixed(2),
            partially_bound: false,
            nonuniform_indexing: true,
        }],
    };
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_sampled_texture_and_storage_buffer_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // The compatibility aggregate must not admit a storage buffer by itself.
    // SAFETY: this mock reports only the compatibility aggregate, not storage-buffer indexing.
    let without = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        without.create_bind_group_layout(&layout),
        Err(Error::Unsupported("nonuniform binding indexing"))
    ));

    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_storage_buffer_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this mock explicitly reports storage-buffer indexing.
    let with = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(with.create_bind_group_layout(&layout).is_ok());
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 1));
}

#[test]
fn sampled_texture_nonuniform_indexing_uses_its_own_capability() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let layout = BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Compute],
            ty: BindingType::SampledTexture {
                sample: TextureSampleType::UnfilterableFloat,
                dimension: TextureViewDimension::D2,
                multisampled: false,
            },
            count: BindingCount::Fixed(2),
            partially_bound: false,
            nonuniform_indexing: true,
        }],
    };
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_sampled_texture_and_storage_buffer_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // The compatibility aggregate must not admit sampled textures by itself.
    // SAFETY: this mock reports only the legacy aggregate, not sampled-texture indexing.
    let without = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        without.create_bind_group_layout(&layout),
        Err(Error::Unsupported("nonuniform binding indexing"))
    ));

    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.nonuniform_sampled_texture_indexing = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this mock explicitly reports sampled-texture indexing.
    let with = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(with.create_bind_group_layout(&layout).is_ok());
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 1));
}

#[test]
fn texture_atomic_layout_is_format_stage_and_capability_scoped() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let layout = |format, visibility| BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: BindingType::StorageTexture {
                format,
                dimension: TextureViewDimension::D2,
                access: StorageAccess::Atomic,
            },
            count: BindingCount::One,
            partially_bound: false,
            nonuniform_indexing: false,
        }],
    };
    let r64_unavailable = device(false);
    assert!(matches!(
        r64_unavailable
            .create_bind_group_layout(&layout(TextureFormat::R64Uint, vec![ShaderStage::Compute])),
        Err(Error::Unsupported("R64Uint compute image atomic binding"))
    ));
    let unsupported = device(false);
    assert!(matches!(
        unsupported
            .create_bind_group_layout(&layout(TextureFormat::R32Uint, vec![ShaderStage::Compute])),
        Err(Error::Unsupported("R32Uint compute texture atomic binding"))
    ));
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.texture_atomic_r32uint_compute = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: the fixture supplies an atomic capability for its mock device.
    let supported = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        supported
            .create_bind_group_layout(&layout(TextureFormat::R32Sint, vec![ShaderStage::Compute])),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        supported.create_bind_group_layout(&layout(
            TextureFormat::R32Uint,
            vec![ShaderStage::Compute, ShaderStage::Fragment]
        )),
        Err(Error::Unsupported(_))
    ));
    assert!(
        supported
            .create_bind_group_layout(&layout(TextureFormat::R32Uint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    assert!(matches!(
        supported
            .create_bind_group_layout(&layout(TextureFormat::R32Sint, vec![ShaderStage::Compute])),
        Err(Error::Unsupported(_))
    ));
    let sint_id = DeviceId::allocate().unwrap();
    let mut sint_caps = DeviceCapabilities::default();
    sint_caps.texture_atomic_r32sint_compute = true;
    sint_caps.limits.max_bindings_per_group = 4;
    // SAFETY: the fixture supplies the signed atomic capability for its mock device.
    let sint_supported =
        unsafe { Device::<NativeShared>::from_native(sint_id, Arc::new(()), sint_caps) };
    let sint_result = sint_supported
        .create_bind_group_layout(&layout(TextureFormat::R32Sint, vec![ShaderStage::Compute]));
    if let Err(error) = sint_result {
        panic!("{error:?}");
    }
    let r64_id = DeviceId::allocate().unwrap();
    let mut r64_caps = DeviceCapabilities::default();
    r64_caps.shader_int64_atomic_image_r64uint_umin = true;
    r64_caps.limits.max_bindings_per_group = 4;
    // SAFETY: the fixture supplies the R64Uint image atomic capability.
    let r64_supported =
        unsafe { Device::<NativeShared>::from_native(r64_id, Arc::new(()), r64_caps) };
    assert!(
        r64_supported
            .create_bind_group_layout(&layout(TextureFormat::R64Uint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    let mut array_image_layout = layout(TextureFormat::R64Uint, vec![ShaderStage::Compute]);
    array_image_layout.entries[0].ty = BindingType::StorageTexture {
        format: TextureFormat::R64Uint,
        dimension: TextureViewDimension::D2Array,
        access: StorageAccess::Atomic,
    };
    assert!(matches!(
        r64_supported.create_bind_group_layout(&array_image_layout),
        Err(Error::Unsupported("R64Uint compute image atomic binding"))
    ));
    let r64_umax_id = DeviceId::allocate().unwrap();
    let mut r64_umax_caps = DeviceCapabilities::default();
    r64_umax_caps.shader_int64_atomic_image_r64uint_umax = true;
    r64_umax_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the R64Uint image unsigned-max route.
    let r64_umax_supported =
        unsafe { Device::<NativeShared>::from_native(r64_umax_id, Arc::new(()), r64_umax_caps) };
    assert!(
        r64_umax_supported
            .create_bind_group_layout(&layout(TextureFormat::R64Uint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    assert!(matches!(
        r64_supported.create_bind_group_layout(&layout(
            TextureFormat::R64Uint,
            vec![ShaderStage::Compute, ShaderStage::Fragment]
        )),
        Err(Error::Unsupported("R64Uint compute image atomic binding"))
    ));
    let min_id = DeviceId::allocate().unwrap();
    let mut min_caps = DeviceCapabilities::default();
    min_caps.texture_atomic_r32uint_min_compute = true;
    min_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the unsigned min atomic route.
    let min_supported =
        unsafe { Device::<NativeShared>::from_native(min_id, Arc::new(()), min_caps) };
    assert!(
        min_supported
            .create_bind_group_layout(&layout(TextureFormat::R32Uint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    let max_id = DeviceId::allocate().unwrap();
    let mut max_caps = DeviceCapabilities::default();
    max_caps.texture_atomic_r32uint_max_compute = true;
    max_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the unsigned max atomic route.
    let max_supported =
        unsafe { Device::<NativeShared>::from_native(max_id, Arc::new(()), max_caps) };
    assert!(
        max_supported
            .create_bind_group_layout(&layout(TextureFormat::R32Uint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    let sint_min_id = DeviceId::allocate().unwrap();
    let mut sint_min_caps = DeviceCapabilities::default();
    sint_min_caps.texture_atomic_r32sint_min_compute = true;
    sint_min_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the signed min atomic route.
    let sint_min_supported =
        unsafe { Device::<NativeShared>::from_native(sint_min_id, Arc::new(()), sint_min_caps) };
    assert!(
        sint_min_supported
            .create_bind_group_layout(&layout(TextureFormat::R32Sint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    let sint_max_id = DeviceId::allocate().unwrap();
    let mut sint_max_caps = DeviceCapabilities::default();
    sint_max_caps.texture_atomic_r32sint_max_compute = true;
    sint_max_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the signed max atomic route.
    let sint_max_supported =
        unsafe { Device::<NativeShared>::from_native(sint_max_id, Arc::new(()), sint_max_caps) };
    assert!(
        sint_max_supported
            .create_bind_group_layout(&layout(TextureFormat::R32Sint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    let sint_min_only_id = DeviceId::allocate().unwrap();
    let mut sint_min_only_caps = DeviceCapabilities::default();
    sint_min_only_caps.texture_atomic_r32sint_min_compute = true;
    sint_min_only_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the signed min route.
    let sint_min_only = unsafe {
        Device::<NativeShared>::from_native(sint_min_only_id, Arc::new(()), sint_min_only_caps)
    };
    assert!(
        sint_min_only
            .create_bind_group_layout(&layout(TextureFormat::R32Sint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    let sint_max_only_id = DeviceId::allocate().unwrap();
    let mut sint_max_only_caps = DeviceCapabilities::default();
    sint_max_only_caps.texture_atomic_r32sint_max_compute = true;
    sint_max_only_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the signed max route.
    let sint_max_only = unsafe {
        Device::<NativeShared>::from_native(sint_max_only_id, Arc::new(()), sint_max_only_caps)
    };
    assert!(
        sint_max_only
            .create_bind_group_layout(&layout(TextureFormat::R32Sint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    let xor_id = DeviceId::allocate().unwrap();
    let mut xor_caps = DeviceCapabilities::default();
    xor_caps.texture_atomic_r32uint_xor_compute = true;
    xor_caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture exposes only the unsigned xor atomic route.
    let xor_supported =
        unsafe { Device::<NativeShared>::from_native(xor_id, Arc::new(()), xor_caps) };
    assert!(
        xor_supported
            .create_bind_group_layout(&layout(TextureFormat::R32Uint, vec![ShaderStage::Compute]))
            .is_ok()
    );
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 11));
}

#[test]
fn acceleration_structure_arrays_require_their_own_capability() {
    LAYOUT_CALLS.with(|calls| calls.set(0));
    let layout = BindGroupLayoutDesc {
        label: None,
        entries: vec![BindGroupLayoutEntry {
            binding: 0,
            visibility: vec![ShaderStage::Compute],
            ty: BindingType::AccelerationStructure,
            count: BindingCount::Fixed(2),
            partially_bound: false,
            nonuniform_indexing: false,
        }],
    };
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this fixture represents a device without the optional ray domain.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        device.create_bind_group_layout(&layout),
        Err(Error::Unsupported("acceleration structure binding arrays"))
    ));
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 0));

    let mut runtime_layout = layout.clone();
    runtime_layout.entries[0].count = BindingCount::RuntimeSized { max_count: 2 };
    let id = DeviceId::allocate().unwrap();
    let mut caps = DeviceCapabilities::default();
    caps.binding_arrays = true;
    caps.runtime_sized_binding_arrays = true;
    caps.limits.max_bindings_per_group = 4;
    // SAFETY: this mock has generic runtime arrays but no acceleration-array route.
    let device = unsafe { Device::<NativeShared>::from_native(id, Arc::new(()), caps) };
    assert!(matches!(
        device.create_bind_group_layout(&runtime_layout),
        Err(Error::Unsupported(
            "runtime-sized acceleration structure arrays"
        ))
    ));
    LAYOUT_CALLS.with(|calls| assert_eq!(calls.get(), 0));
}

#[test]
fn runtime_array_group_validates_actual_length_and_holes_before_backend() {
    GROUP_CALLS.with(|calls| calls.set(0));
    let device = device(true);
    let make_layout = |partially_bound| {
        device
            .create_bind_group_layout(&BindGroupLayoutDesc {
                label: None,
                entries: vec![entry(
                    0,
                    BindingCount::RuntimeSized { max_count: 2 },
                    partially_bound,
                )],
            })
            .unwrap()
    };
    let full = make_layout(false);
    let partial = make_layout(true);
    // SAFETY: the test sampler is a mock native handle owned by this device.
    let sampler = unsafe { Sampler::<NativeShared>::from_native(device.id(), Arc::new(())) };
    let binding = || Some(BindingResource::Sampler(&sampler));
    let create = |layout: &BindGroupLayout<NativeShared>, values| {
        let entries = [BindGroupEntry {
            binding: 0,
            resource: BindingResource::Array(values),
        }];
        device.create_bind_group(&BindGroupDesc {
            label: None,
            layout,
            entries: &entries,
        })
    };
    assert!(matches!(create(&full, vec![]), Err(Error::InvalidState)));
    assert!(matches!(
        create(&full, vec![binding(), binding(), binding()]),
        Err(Error::InvalidRange)
    ));
    assert!(matches!(
        create(&full, vec![binding(), None]),
        Err(Error::InvalidState)
    ));
    assert!(create(&full, vec![binding()]).is_ok());
    assert!(create(&full, vec![binding(), binding()]).is_ok());
    assert!(create(&partial, vec![binding(), None]).is_ok());
    GROUP_CALLS.with(|calls| assert_eq!(calls.get(), 3));
}

#[test]
fn arrays_reject_nested_descriptors_without_losing_leaf_validation() {
    GROUP_CALLS.with(|calls| calls.set(0));
    let device = device(true);
    let foreign = self::device(true);
    // SAFETY: each mock handle belongs to its associated fixture device.
    let sampler = unsafe { Sampler::<NativeShared>::from_native(device.id(), Arc::new(())) };
    // SAFETY: the mock handle is deliberately owned by the other fixture device.
    let foreign_sampler =
        unsafe { Sampler::<NativeShared>::from_native(foreign.id(), Arc::new(())) };
    // SAFETY: the mock buffer belongs to this fixture device.
    let buffer = unsafe { Buffer::<NativeShared>::from_native(device.id(), Arc::new(())) };
    for count in [
        BindingCount::Fixed(1),
        BindingCount::RuntimeSized { max_count: 2 },
    ] {
        // SAFETY: this test imports matching layout metadata to isolate group validation.
        let layout = unsafe {
            BindGroupLayout::<NativeShared>::from_native(
                device.id(),
                Arc::new(()),
                Arc::from([entry(0, count, true)]),
            )
        };
        let create = |values| {
            let entries = [BindGroupEntry {
                binding: 0,
                resource: BindingResource::Array(values),
            }];
            device.create_bind_group(&BindGroupDesc {
                label: None,
                layout: &layout,
                entries: &entries,
            })
        };
        for inner in [
            vec![],
            vec![None],
            vec![Some(BindingResource::Sampler(&sampler))],
        ] {
            assert!(matches!(
                create(vec![Some(BindingResource::Array(inner))]),
                Err(Error::InvalidState)
            ));
        }
        assert!(matches!(
            create(vec![Some(BindingResource::Sampler(&foreign_sampler))]),
            Err(Error::WrongDevice)
        ));
        assert!(matches!(
            create(vec![Some(BindingResource::Buffer {
                buffer: &buffer,
                range: 0..4
            })]),
            Err(Error::InvalidState)
        ));
        GROUP_CALLS.with(|calls| assert_eq!(calls.get(), 0));
    }
}

#[test]
fn storage_views_validate_abi_range_kind_and_device_before_backend() {
    GROUP_CALLS.with(|calls| calls.set(0));
    let device = device(false);
    let foreign = self::device(false);
    // SAFETY: the mock handles belong to their respective devices.
    let buffer = unsafe { Buffer::<NativeShared>::from_native(device.id(), Arc::new(())) };
    let other = unsafe { Buffer::<NativeShared>::from_native(foreign.id(), Arc::new(())) };
    let mut slot = entry(0, BindingCount::One, false);
    slot.ty = BindingType::StorageBuffer {
        dynamic_offset: false,
        min_size: 8,
        access: StorageAccess::Read,
    };
    // SAFETY: this mock layout carries the matching fixture device identity.
    let layout = unsafe {
        BindGroupLayout::<NativeShared>::from_native(device.id(), Arc::new(()), Arc::from([slot]))
    };
    let create = |buffer, range, view| {
        let entries = [BindGroupEntry {
            binding: 0,
            resource: BindingResource::StorageBufferView {
                buffer,
                range,
                view,
            },
        }];
        device.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &entries,
        })
    };
    assert!(create(&buffer, 0..16, StorageBufferView::Raw).is_ok());
    assert!(
        create(
            &buffer,
            16..48,
            StorageBufferView::Structured { stride: 16 }
        )
        .is_ok()
    );
    for (range, view) in [
        (0..0, StorageBufferView::Raw),
        (4..8, StorageBufferView::Raw),
        (2..10, StorageBufferView::Raw),
        (0..16, StorageBufferView::Structured { stride: 0 }),
        (0..16, StorageBufferView::Structured { stride: 6 }),
        (4..20, StorageBufferView::Structured { stride: 16 }),
    ] {
        assert!(matches!(
            create(&buffer, range, view),
            Err(Error::InvalidState)
        ));
    }
    assert!(matches!(
        create(&other, 0..16, StorageBufferView::Raw),
        Err(Error::WrongDevice)
    ));
    assert!(!binding_matches(
        &BindingType::UniformBuffer {
            dynamic_offset: false,
            min_size: 8
        },
        &BindingResource::StorageBufferView {
            buffer: &buffer,
            range: 0..16,
            view: StorageBufferView::Raw,
        },
    ));
    GROUP_CALLS.with(|calls| assert_eq!(calls.get(), 2));
}
