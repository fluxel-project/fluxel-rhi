//! Runtime array lengths, holes, and nested descriptor rejection.
use super::{GROUP_CALLS, LAYOUT_CALLS, TEXTURE_CALLS, device, entry};
use crate::api::tests::type_contract_tests::NativeShared;
use crate::{api::*, backend::*};
use std::sync::Arc;

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
