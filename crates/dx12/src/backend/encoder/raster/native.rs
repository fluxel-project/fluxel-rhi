//! Native D3D12 primitives shared by raster and upload recording.
//!
//! These helpers own only temporary FFI descriptors.  The caller retains every
//! actual resource in its encoder until fence retirement; this module creates
//! neither render passes nor resource-state history.

use windows::Win32::Graphics::Direct3D12::*;

pub(crate) fn barrier(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    super::super::transition::barrier_sub(
        list,
        resource,
        D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
        before,
        after,
    );
}
