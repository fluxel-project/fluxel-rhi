//! Shared-resource and fence boundaries execute in exact native queue order.
#![cfg(all(windows, feature = "hardware-tests"))]

use fluxel_rhi_core::api::*;
use fluxel_rhi_dx12::{Dx12ExternalHandle, Dx12Instance};
use std::{
    os::windows::io::{FromRawHandle, OwnedHandle},
    sync::Arc,
};
use windows::{
    Win32::{
        Foundation::{GENERIC_ALL, HANDLE},
        Graphics::{Direct3D12::*, Dxgi::Common::*},
    },
    core::{Interface, PCWSTR},
};

#[path = "support/mod.rs"]
mod support;
use support::block;

fn native_device(device: &Device<fluxel_rhi_dx12::Dx12>) -> ID3D12Device {
    let domain = device.native_interop().unwrap();
    // SAFETY: the guard holds the RHI-owned device interface while we clone a
    // COM reference for this test's independent producer allocation.
    unsafe {
        let guard = domain.acquire(NativeInteropRequest::Device).unwrap();
        ID3D12Device::from_raw_borrowed(&guard.raw_pointer())
            .unwrap()
            .clone()
    }
}

unsafe fn owned(handle: HANDLE) -> OwnedHandle {
    // SAFETY: every call site receives one owned handle from CreateSharedHandle.
    unsafe { OwnedHandle::from_raw_handle(handle.0) }
}

fn submit(
    device: &Device<fluxel_rhi_dx12::Dx12>,
    lane: LaneId,
    ordinal: u64,
    work: RecordedWork<fluxel_rhi_dx12::Dx12>,
) -> CompletionPoint<fluxel_rhi_dx12::Dx12> {
    let plan = unsafe {
        SubmissionPlan::from_compiled(
            PlanId {
                device: device.id(),
                ordinal,
            },
            Arc::from([StepDesc {
                lane,
                kind: StepKind::Commands,
                waits: vec![],
            }]),
        )
    };
    let step = plan.step(0);
    let outcome = unsafe {
        device.queue().submit(SubmitPacket {
            plan,
            steps: vec![BoundStep {
                step,
                payload: StepPayload::Commands(vec![work]),
            }],
            completion: CompletionRequest::End,
        })
    };
    match outcome {
        SubmitOutcome::Accepted(receipt) => receipt.checkpoints.into_iter().next().unwrap().point,
        _ => panic!("external submission failed"),
    }
}

#[test]
fn shared_resource_fence_boundaries_cross_independent_devices() {
    let producer = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let importer = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let native = native_device(&producer);
    let native_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 16,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let properties = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let mut shared_resource: Option<ID3D12Resource> = None;
    // SAFETY: this producer resource has a complete shared default-heap descriptor.
    unsafe {
        native.CreateCommittedResource(
            &properties,
            D3D12_HEAP_FLAG_SHARED,
            &native_desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut shared_resource,
        )
    }
    .unwrap();
    let resource_handle = unsafe {
        native.CreateSharedHandle(
            &shared_resource.unwrap(),
            None,
            GENERIC_ALL.0,
            PCWSTR::null(),
        )
    }
    .unwrap();
    let producer_fence =
        unsafe { native.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_SHARED) }.unwrap();
    let initial_fence_handle =
        unsafe { native.CreateSharedHandle(&producer_fence, None, GENERIC_ALL.0, PCWSTR::null()) }
            .unwrap();

    let external = importer.external().unwrap();
    let reserved_fence_handle =
        unsafe { native.CreateSharedHandle(&producer_fence, None, GENERIC_ALL.0, PCWSTR::null()) }
            .unwrap();
    assert!(matches!(
        unsafe {
            external.import_sync(Dx12ExternalHandle::Fence {
                handle: owned(reserved_fence_handle),
                value: u64::MAX,
            })
        },
        Err(Error::InvalidRange)
    ));
    let memory = unsafe {
        external.import_memory(
            Dx12ExternalHandle::Resource(owned(resource_handle)),
            &ExternalResourceDesc {
                size: 16,
                offset: 0,
                usage: BufferUsage::COPY_SRC | BufferUsage::COPY_DST,
            },
        )
    }
    .unwrap();
    let shared = external.buffer(&memory).unwrap();
    let initial = unsafe {
        external.import_sync(Dx12ExternalHandle::Fence {
            handle: owned(initial_fence_handle),
            value: 0,
        })
    }
    .unwrap();
    let lane = importer
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let before = importer
        .create_buffer(&BufferDesc {
            label: None,
            size: 4,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let after = importer
        .create_buffer(&BufferDesc {
            label: None,
            size: 4,
            usage: BufferUsage::COPY_DST,
            memory: MemoryClass::Device,
        })
        .unwrap();
    let expected = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88];
    let mut first = importer.encoder(lane).unwrap();
    // This command must remain before the imported queue wait.
    first.write_buffer(&before, 0, &[1, 2, 3, 4]).unwrap();
    external.acquire(&mut first, &memory, initial).unwrap();
    // This command is recorded between the wait and exported signal.
    first.write_buffer(&shared, 0, &expected).unwrap();
    let exported = external.release(&mut first, &memory).unwrap();
    // This command must remain after the exported queue signal.
    first.write_buffer(&after, 0, &[5, 6, 7, 8]).unwrap();
    let exported_handle = exported.into_native().into_handle().unwrap();
    let point = submit(&importer, lane, 9_901, first.finish().unwrap());
    block(importer.wait_completion(&point)).unwrap();

    let Dx12ExternalHandle::Fence {
        handle: exported_handle,
        value,
    } = exported_handle
    else {
        unreachable!()
    };
    let mut observed = None;
    // SAFETY: this opens the importer-exported shared fence on the independent
    // producer device; the fence handle remains owned until this call returns.
    unsafe {
        native.OpenSharedHandle::<ID3D12Fence>(
            HANDLE(std::os::windows::io::AsRawHandle::as_raw_handle(&exported_handle).cast()),
            &mut observed,
        )
    }
    .unwrap();
    assert!(unsafe { observed.unwrap().GetCompletedValue() } >= value);

    let acquired = unsafe {
        external.import_sync(Dx12ExternalHandle::Fence {
            handle: exported_handle,
            value,
        })
    }
    .unwrap();
    let readback = importer
        .create_buffer(&BufferDesc {
            label: None,
            size: expected.len() as u64,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let empty = BufferUses {
        stages: StageMask::EMPTY,
        accesses: AccessMask::EMPTY,
    };
    let copy_read = BufferUses {
        stages: StageMask::COPY,
        accesses: AccessMask::COPY_READ,
    };
    let mut second = importer.encoder(lane).unwrap();
    external.acquire(&mut second, &memory, acquired).unwrap();
    second
        .transition_resources(&[ResourceTransition::Buffer(BufferBarrier {
            target: BufferSlice {
                buffer: &shared,
                range: 0..16,
            },
            usage: StateTransition {
                from: empty,
                to: copy_read,
            },
            ownership: None,
        })])
        .unwrap();
    second
        .copy_buffer(&shared, 0, &readback, 0, expected.len() as u64)
        .unwrap();
    let ticket = second
        .readback_buffer(&readback, 0..expected.len() as u64)
        .unwrap();
    let _ = external.release(&mut second, &memory).unwrap();
    let point = submit(&importer, lane, 9_902, second.finish().unwrap());
    block(importer.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(importer.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(bytes, expected);
}

#[test]
fn rejects_storage_import_without_native_uav_capability() {
    let producer = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let importer = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let native = native_device(&producer);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: 16,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
        ..Default::default()
    };
    let properties = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: this is a valid shared default resource intentionally lacking UAV capability.
    unsafe {
        native.CreateCommittedResource(
            &properties,
            D3D12_HEAP_FLAG_SHARED,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )
    }
    .unwrap();
    let handle = unsafe {
        native.CreateSharedHandle(&resource.unwrap(), None, GENERIC_ALL.0, PCWSTR::null())
    }
    .unwrap();
    let external = importer.external().unwrap();
    let result = unsafe {
        external.import_memory(
            Dx12ExternalHandle::Resource(owned(handle)),
            &ExternalResourceDesc {
                size: 16,
                offset: 0,
                usage: BufferUsage::STORAGE,
            },
        )
    };
    assert!(matches!(result, Err(Error::InvalidState)));
}

#[test]
fn shared_texture_roundtrips_through_external_layout_boundaries() {
    let producer = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let importer = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let native = native_device(&producer);
    let native_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 2,
        Height: 2,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        ..Default::default()
    };
    let properties = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: complete shared texture resource descriptor on the producer device.
    unsafe {
        native.CreateCommittedResource(
            &properties,
            D3D12_HEAP_FLAG_SHARED,
            &native_desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )
    }
    .unwrap();
    let resource_handle = unsafe {
        native.CreateSharedHandle(&resource.unwrap(), None, GENERIC_ALL.0, PCWSTR::null())
    }
    .unwrap();
    let initial_fence =
        unsafe { native.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_SHARED) }.unwrap();
    let initial_fence_handle =
        unsafe { native.CreateSharedHandle(&initial_fence, None, GENERIC_ALL.0, PCWSTR::null()) }
            .unwrap();
    let texture_desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [2, 2, 1],
        array_layers: 1,
        format: TextureFormat::Rgba8Unorm,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
        view_formats: vec![],
    };
    let transition = ExternalTextureTransition {
        aspects: TextureAspects::COLOR,
        mips: 0..1,
        layers: 0..1,
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
            to: TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
        },
        layout: (ImageLayout::Common, ImageLayout::Common),
    };
    let external = importer.external().unwrap();
    let texture = unsafe {
        external.import_texture(
            Dx12ExternalHandle::Resource(owned(resource_handle)),
            &texture_desc,
        )
    }
    .unwrap();
    let shared = external.texture(&texture).unwrap();
    let initial = unsafe {
        external.import_sync(Dx12ExternalHandle::Fence {
            handle: owned(initial_fence_handle),
            value: 0,
        })
    }
    .unwrap();
    let lane = importer
        .lanes()
        .iter()
        .find(|lane| lane.class == LaneClass::Graphics)
        .unwrap()
        .id;
    let mut upload = vec![0u8; 512];
    upload[..8].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);
    upload[256..264].copy_from_slice(&[9, 10, 11, 12, 13, 14, 15, 16]);
    let mut first = importer.encoder(lane).unwrap();
    external
        .acquire_texture(&mut first, &texture, initial, &transition)
        .unwrap();
    first
        .write_texture(
            TextureRegion {
                texture: &shared,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 2, 1],
                aspect: TextureAspects::COLOR,
            },
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
            &upload,
        )
        .unwrap();
    let exported = external
        .release_texture(&mut first, &texture, &transition)
        .unwrap();
    let exported = exported.into_native().into_handle().unwrap();
    let point = submit(&importer, lane, 9_903, first.finish().unwrap());
    block(importer.wait_completion(&point)).unwrap();

    let Dx12ExternalHandle::Fence { handle, value } = exported else {
        unreachable!()
    };
    let sync =
        unsafe { external.import_sync(Dx12ExternalHandle::Fence { handle, value }) }.unwrap();
    let readback = importer
        .create_buffer(&BufferDesc {
            label: None,
            size: 512,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            memory: MemoryClass::Readback,
        })
        .unwrap();
    let mut second = importer.encoder(lane).unwrap();
    external
        .acquire_texture(&mut second, &texture, sync, &transition)
        .unwrap();
    second
        .transition_resources(&[ResourceTransition::Texture(TextureBarrier {
            target: Subresources {
                texture: &shared,
                aspects: TextureAspects::COLOR,
                mips: 0..1,
                layers: 0..1,
            },
            usage: StateTransition {
                from: TextureUses {
                    stages: StageMask::EMPTY,
                    accesses: AccessMask::EMPTY,
                },
                to: TextureUses {
                    stages: StageMask::COPY,
                    accesses: AccessMask::COPY_READ,
                },
            },
            layout: Some((ImageLayout::Common, ImageLayout::CopySrc)),
            ownership: None,
        })])
        .unwrap();
    second
        .texture_to_buffer(
            TextureRegion {
                texture: &shared,
                mip: 0,
                origin: [0, 0, 0],
                extent: [2, 2, 1],
                aspect: TextureAspects::COLOR,
            },
            &readback,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: 256,
                rows_per_image: 2,
            },
        )
        .unwrap();
    let ticket = second.readback_buffer(&readback, 0..512).unwrap();
    let release_transition = ExternalTextureTransition {
        layout: (ImageLayout::CopySrc, ImageLayout::Common),
        usage: StateTransition {
            from: TextureUses {
                stages: StageMask::COPY,
                accesses: AccessMask::COPY_READ,
            },
            to: TextureUses {
                stages: StageMask::EMPTY,
                accesses: AccessMask::EMPTY,
            },
        },
        ..transition
    };
    let _ = external
        .release_texture(&mut second, &texture, &release_transition)
        .unwrap();
    let point = submit(&importer, lane, 9_904, second.finish().unwrap());
    block(importer.wait_completion(&point)).unwrap();
    let bytes = unsafe { block(importer.read_after(ticket, &point)).unwrap().bytes };
    assert_eq!(&bytes[..8], &upload[..8]);
    assert_eq!(&bytes[256..264], &upload[256..264]);
}

#[test]
fn imports_typeless_depth_texture_for_sampled_core_descriptor() {
    let producer = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let importer = block(Dx12Instance::new().unwrap().open_default()).unwrap();
    let native = native_device(&producer);
    let resource_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 4,
        Height: 4,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32_TYPELESS,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
        ..Default::default()
    };
    let properties = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let mut resource: Option<ID3D12Resource> = None;
    // SAFETY: a shared typeless depth resource has the same allocation form
    // that the DX12 backend uses for sampled Depth32Float textures.
    unsafe {
        native.CreateCommittedResource(
            &properties,
            D3D12_HEAP_FLAG_SHARED,
            &resource_desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,
            None,
            &mut resource,
        )
    }
    .unwrap();
    let handle = unsafe {
        native.CreateSharedHandle(&resource.unwrap(), None, GENERIC_ALL.0, PCWSTR::null())
    }
    .unwrap();
    let desc = TextureDesc {
        label: None,
        dimension: TextureDimension::D2,
        extent: [4, 4, 1],
        array_layers: 1,
        format: TextureFormat::Depth32Float,
        mip_levels: 1,
        samples: 1,
        usage: TextureUsage::TEXTURE_BINDING | TextureUsage::RENDER_ATTACHMENT,
        view_formats: vec![],
    };
    let domain = importer.external().unwrap();
    let texture =
        unsafe { domain.import_texture(Dx12ExternalHandle::Resource(owned(handle)), &desc) }
            .unwrap();
    assert!(domain.texture(&texture).is_ok());
}
