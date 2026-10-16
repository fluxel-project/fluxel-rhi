//! Bind-group resource validation used while writing descriptor-table ranges.
//!
//! This module validates only the resource-to-layout correspondence and never
//! allocates heaps or writes descriptors.  Keeping it separate makes the
//! parent descriptor module own heap ranges and bind-group lifetime, while the
//! writer path owns the per-entry runtime-array and view-dimension contract.

use fluxel_rhi_core::api::{self, Error, Result};

use super::{BindingSlot, Dx12};

pub(super) fn resource_descriptor_count(
    slot: &BindingSlot,
    resource: &api::BindingResource<'_, Dx12>,
) -> Result<u32> {
    if !slot.runtime_sized {
        return Ok(slot.descriptor_count);
    }
    let api::BindingResource::Array(values) = resource else {
        return Err(Error::InvalidState);
    };
    let count = u32::try_from(values.len()).map_err(|_| Error::InvalidRange)?;
    if count == 0 || count > slot.descriptor_count {
        return Err(Error::InvalidRange);
    }
    Ok(count)
}

pub(super) fn validate_sampled_view_dimension(
    slot: &BindingSlot,
    resource: &api::BindingResource<'_, Dx12>,
) -> Result<()> {
    let expected = slot.view_dimension.ok_or(Error::InvalidState)?;
    match resource {
        api::BindingResource::TextureView(view) => validate_sampled_view(view, expected),
        api::BindingResource::Array(views) => {
            for view in views.iter().flatten() {
                let api::BindingResource::TextureView(view) = view else {
                    return Err(Error::Unsupported("DX12 sampled texture array resource"));
                };
                validate_sampled_view(view, expected)?;
            }
            Ok(())
        }
        _ => Err(Error::Unsupported("DX12 sampled texture binding resource")),
    }
}

fn validate_sampled_view(
    view: &api::TextureView<Dx12>,
    expected: api::TextureViewDimension,
) -> Result<()> {
    // SAFETY: Core checked this view's device identity; read its immutable
    // Rust-side descriptor only, without exposing native handles.
    if unsafe { view.native_ref() }.0.desc.dimension != expected {
        return Err(Error::InvalidState);
    }
    Ok(())
}
