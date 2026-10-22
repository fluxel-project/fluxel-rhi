//! Install the device-wide shader-visible heaps on each fresh command list.
//!
//! Every bind group and internal blit uses suballocations of these same heaps.
//! The list starts with no heap bindings after creation or reset; bundles must
//! declare the same heaps as their parent before setting descriptor tables.
use super::*;

pub(super) fn install(device: &Dx12Device, list: &ID3D12GraphicsCommandList) {
    let heaps = [
        Some(device.0.descriptors.view_heap().clone()),
        Some(device.0.descriptors.sampler_heap().clone()),
    ];
    crate::diagnostics::record(crate::diagnostics::Operation::DescriptorHeapSets, 1);
    // SAFETY: both heaps belong to the list's device and are retained by it.
    unsafe { list.SetDescriptorHeaps(&heaps) };
}
