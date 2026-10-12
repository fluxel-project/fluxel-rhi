//! DX12 bounded runtime texture-array descriptors.
//!
//! D3D12 descriptor-table ranges are fixed when the root signature is created.
//! The public runtime-sized binding in example 14 is therefore bounded to the
//! advertised 32 descriptors. A bind group supplies a contiguous, nonempty
//! prefix and retains every view written into that range.

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

use super::super::{Dx12Device, Dx12TextureView};

pub(super) const RUNTIME_SRV_CAPACITY: u32 = 32;

pub(super) fn count(count: api::BindingCount) -> Result<u32> {
    match count {
        api::BindingCount::One => Ok(1),
        api::BindingCount::Fixed(value) if value > 0 => Ok(value),
        api::BindingCount::Fixed(_) => Err(Error::InvalidRange),
        api::BindingCount::RuntimeSized => Ok(RUNTIME_SRV_CAPACITY),
    }
}

pub(super) fn write_sampled_array(
    device: &Dx12Device,
    destination: u32,
    expected: u32,
    values: &[Option<api::BindingResource<'_, super::super::Dx12>>],
) -> Result<Vec<Dx12TextureView>> {
    if values.len() != expected as usize || values.iter().any(Option::is_none) {
        return Err(Error::InvalidState);
    }
    let mut retained = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let Some(api::BindingResource::TextureView(view)) = value else {
            return Err(Error::Unsupported("DX12 sampled texture array resource"));
        };
        let native = unsafe { view.native_ref() };
        let source = native
            .0
            .srv
            .ok_or(Error::Unsupported("DX12 sampled texture without SRV"))?;
        unsafe {
            device.0.native.CopyDescriptorsSimple(
                1,
                device.0.descriptors.views.cpu(destination + index as u32),
                source,
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            );
        }
        retained.push(native.clone());
    }
    Ok(retained)
}
