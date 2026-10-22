//! Core validation of runtime-sized binding array layouts and instances.
use super::{type_contract_tests::NativeShared, *};
use std::cell::Cell;

thread_local! {
    static LAYOUT_CALLS: Cell<usize> = const { Cell::new(0) };
    static GROUP_CALLS: Cell<usize> = const { Cell::new(0) };
}

impl ResourceBackend for NativeShared {
    fn buffer_support(_: &Arc<()>, _: &BufferDesc) -> Support<BufferSupport> {
        unreachable!()
    }
    fn texture_support(_: &Arc<()>, _: &TextureDesc) -> Support<TextureSupport> {
        unreachable!()
    }
    fn create_buffer(_: &Arc<()>, _: &BufferDesc) -> Result<Arc<()>> {
        unreachable!()
    }
    fn create_texture(_: &Arc<()>, _: &TextureDesc) -> Result<Arc<()>> {
        unreachable!()
    }
    fn create_view(_: &Arc<()>, _: &Arc<()>, _: &TextureViewDesc) -> Result<Arc<()>> {
        unreachable!()
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
