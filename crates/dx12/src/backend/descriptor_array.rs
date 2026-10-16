//! DX12 bounded runtime texture-array descriptors.
//!
//! D3D12 descriptor-table ranges are fixed when the root signature is created.
//! The public runtime-sized binding carries its explicit layout capacity.
//! A bind group supplies a contiguous, nonempty
//! prefix and retains every view written into that range. The parent bind group
//! owns the retained views, so the copied descriptors cannot outlive their
//! native resources while GPU work still refers to them.

use fluxel_rhi_core::api::{self, Error, Result};
use windows::Win32::Graphics::Direct3D12::D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;

use super::super::{Dx12Device, Dx12TextureView};

pub(super) fn count(count: api::BindingCount) -> Result<u32> {
    match count {
        api::BindingCount::One => Ok(1),
        api::BindingCount::Fixed(value) if value > 0 => Ok(value),
        api::BindingCount::Fixed(_) => Err(Error::InvalidRange),
        api::BindingCount::RuntimeSized { max_count } if max_count > 0 => Ok(max_count),
        api::BindingCount::RuntimeSized { .. } => Err(Error::InvalidRange),
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
        // SAFETY: Core's typed handle check admits only a view from this DX12
        // device. The cloned view below keeps the source descriptor's resource
        // alive after this short borrow ends.
        let native = unsafe { view.native_ref() };
        let source = native
            .0
            .srv
            .ok_or(Error::Unsupported("DX12 sampled texture without SRV"))?;
        // SAFETY: the destination is a distinct, reserved entry in the live
        // device heap; `source` names one descriptor in a live CPU heap.
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
