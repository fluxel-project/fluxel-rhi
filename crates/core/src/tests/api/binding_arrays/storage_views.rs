//! Storage-buffer view ABI, range, kind, and device validation.
use super::{GROUP_CALLS, LAYOUT_CALLS, TEXTURE_CALLS, device, entry};
use crate::api::tests::type_contract_tests::NativeShared;
use crate::{api::*, backend::*};
use std::sync::Arc;

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
