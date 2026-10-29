//! Failed storage-view creation must release DX12 descriptor reservations.

use super::*;

#[test]
fn failed_storage_views_release_descriptor_reservations() {
    let d = block_on(Dx12Instance::new().unwrap().open_default()).unwrap();
    let buffer = d
        .create_buffer(&BufferDesc {
            label: None,
            size: 2052,
            usage: BufferUsage::STORAGE,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let layout = d
        .create_bind_group_layout(&BindGroupLayoutDesc {
            label: None,
            entries: vec![BindGroupLayoutEntry {
                binding: 0,
                visibility: vec![ShaderStage::Mesh],
                ty: BindingType::StorageBuffer {
                    dynamic_offset: false,
                    min_size: 4,
                    access: StorageAccess::Read,
                },
                count: BindingCount::One,
                partially_bound: false,
                nonuniform_indexing: false,
            }],
        })
        .unwrap();
    for _ in 0..65_537 {
        let entries = [BindGroupEntry {
            binding: 0,
            resource: BindingResource::StorageBufferView {
                buffer: &buffer,
                range: 0..2052,
                view: StorageBufferView::Structured { stride: 2052 },
            },
        }];
        assert!(
            d.create_bind_group(&BindGroupDesc {
                label: None,
                layout: &layout,
                entries: &entries
            })
            .is_err()
        );
    }
    let entries = [BindGroupEntry {
        binding: 0,
        resource: BindingResource::StorageBufferView {
            buffer: &buffer,
            range: 0..2052,
            view: StorageBufferView::Raw,
        },
    }];
    assert!(
        d.create_bind_group(&BindGroupDesc {
            label: None,
            layout: &layout,
            entries: &entries
        })
        .is_ok()
    );
}
