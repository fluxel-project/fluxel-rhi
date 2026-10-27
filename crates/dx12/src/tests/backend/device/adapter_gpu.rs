//! Hardware fixture for the native DXGI adapter identity boundary.
//!
//! This is deliberately separate from the portable device tests: it creates
//! native D3D12 devices and therefore requires a real driver. It verifies that
//! the Core-facing adapter list remains a lossless view of DXGI enumeration.

use super::*;
use std::{
    future::Future,
    task::{Context, Poll, Waker},
};
use windows::Win32::{
    Foundation::LUID,
    Graphics::{
        Direct3D12::{
            D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_FEATURE_D3D12_OPTIONS16,
            D3D12_FEATURE_DATA_D3D12_OPTIONS16, ID3D12CommandAllocator, ID3D12GraphicsCommandList,
            ID3D12GraphicsCommandList9,
        },
        Dxgi::DXGI_ERROR_NOT_FOUND,
    },
};

struct NativeAdapter {
    name: String,
    luid: LUID,
}

fn enumerate_native(instance: &Dx12Instance) -> Result<Vec<NativeAdapter>> {
    let mut adapters = Vec::new();
    for ordinal in 0..=u32::MAX {
        // SAFETY: `instance` owns the factory and this fixture keeps each
        // returned adapter alive until its description has been copied.
        match unsafe { instance.factory.EnumAdapters1(ordinal) } {
            Ok(adapter) => {
                // SAFETY: `adapter` is live and DXGI initializes the returned
                // descriptor synchronously without retaining Rust memory.
                let desc = unsafe { adapter.GetDesc1() }.map_err(native_error)?;
                adapters.push(NativeAdapter {
                    name: adapter_name(&adapter)?,
                    luid: desc.AdapterLuid,
                });
            }
            Err(error) if error.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(error) => return Err(native_error(error)),
        }
    }
    Ok(adapters)
}

fn block_on_ready<T>(future: impl Future<Output = Result<T>>) -> Result<T> {
    let mut future = std::pin::pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut context) {
        Poll::Ready(result) => result,
        Poll::Pending => panic!("DX12 device opening must be immediately ready"),
    }
}

#[test]
fn core_adapters_match_dxgi_and_open_on_their_native_luids() -> Result<()> {
    let native_instance = Dx12Instance::new()?;
    let instance = native_instance.instance()?;
    let native_adapters = enumerate_native(&native_instance)?;
    let adapters = instance.enumerate()?;

    assert_eq!(
        adapters.len(),
        native_adapters.len(),
        "Core adapter enumeration must expose every DXGI ordinal"
    );
    for (ordinal, (adapter, native)) in adapters.iter().zip(&native_adapters).enumerate() {
        assert_eq!(adapter.id.ordinal, ordinal as u64);
        assert_eq!(adapter.name, native.name);
    }

    if adapters.is_empty() {
        eprintln!("SKIP: DXGI reported no adapters; no D3D12 device can be opened");
        return Ok(());
    }

    let mut opened = 0;
    for (adapter, native) in adapters.iter().zip(&native_adapters) {
        let request = DeviceRequest {
            adapter: adapter.id,
            required_features: Vec::new(),
        };
        match block_on_ready(<Dx12 as InstanceBackend>::open(
            &native_instance,
            DeviceId::allocate()?,
            &request,
        )) {
            Ok(device) => {
                // SAFETY: `device` owns a live ID3D12Device and GetAdapterLuid
                // returns its value without retaining any Rust references.
                let actual_luid = unsafe { device.0.0.native.GetAdapterLuid() };
                assert_eq!(
                    actual_luid, native.luid,
                    "adapter ordinal {}",
                    adapter.id.ordinal
                );
                eprintln!(
                    "opened DXGI ordinal {}: {}; matching LUID {actual_luid:?}",
                    adapter.id.ordinal, adapter.name
                );
                opened += 1;
            }
            Err(Error::Unsupported(reason)) => {
                eprintln!(
                    "SKIP: DXGI adapter ordinal {} does not support D3D12: {reason}",
                    adapter.id.ordinal
                );
            }
            Err(error) => return Err(error),
        }
    }

    if opened == 0 {
        eprintln!("SKIP: no DXGI adapter on this machine supports D3D12");
    }
    Ok(())
}

#[test]
fn dynamic_depth_bias_capability_matches_options16_and_command_list9() -> Result<()> {
    let native_instance = Dx12Instance::new()?;
    let instance = native_instance.instance()?;
    let Some(adapter) = instance.enumerate()?.into_iter().next() else {
        eprintln!("SKIP: DXGI reported no adapters");
        return Ok(());
    };
    let request = DeviceRequest {
        adapter: adapter.id,
        required_features: Vec::new(),
    };
    let Ok((device, caps)) = block_on_ready(<Dx12 as InstanceBackend>::open(
        &native_instance,
        DeviceId::allocate()?,
        &request,
    )) else {
        eprintln!("SKIP: selected DXGI adapter does not support D3D12");
        return Ok(());
    };
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    // SAFETY: the live device receives the exact OPTIONS16 structure and the
    // synchronous query retains neither the output nor caller memory.
    let options16 = unsafe {
        device.0.native.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS16,
            (&mut options as *mut _) as *mut _,
            std::mem::size_of_val(&options) as u32,
        )
    }
    .is_ok()
        && options.DynamicDepthBiasSupported.as_bool();
    assert_eq!(caps.dynamic_depth_bias, options16);
    assert_eq!(caps.dynamic_depth_bias_clamp, options16);
    if options16 {
        // SAFETY: the returned allocator/list belong to this live device and
        // are used only to validate the interface required by lowering.
        let allocator = unsafe {
            device
                .0
                .native
                .CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT)
        }
        .map_err(native_error)?;
        let list: ID3D12GraphicsCommandList = unsafe {
            device
                .0
                .native
                .CreateCommandList::<_, _, ID3D12GraphicsCommandList>(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocator,
                    None,
                )
        }
        .map_err(native_error)?;
        assert!(list.cast::<ID3D12GraphicsCommandList9>().is_ok());
    }
    Ok(())
}
