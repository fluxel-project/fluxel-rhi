//! Resolve active view subresources and finish the raster pass.
use super::*;

pub(in crate::backend::encoder::raster) fn end(
    encoder: &mut Dx12Encoder,
    state: RasterState,
) -> Result<(), Error> {
    for target in state.targets {
        if let Some(resolve) = target.resolve {
            let destination_before = if resolve.surface {
                D3D12_RESOURCE_STATE_PRESENT
            } else {
                D3D12_RESOURCE_STATE_RESOLVE_DEST
            };
            for layer in 0..target.layer_count {
                if !selected(state.multiview_mask, layer) {
                    continue;
                }
                let source_subresource = target.subresource + layer * target.mip_levels;
                let destination_subresource = resolve.subresource + layer * resolve.mip_levels;
                super::super::super::transition::barrier_sub(
                    &encoder.list,
                    &target.resource,
                    source_subresource,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                );
                super::super::super::transition::barrier_sub(
                    &encoder.list,
                    &resolve.resource,
                    destination_subresource,
                    destination_before,
                    D3D12_RESOURCE_STATE_RESOLVE_DEST,
                );
                // SAFETY: the pass validated both subresources, view formats,
                // and the render area against the matching attachment extents.
                unsafe {
                    if let Some((list, rect)) = &state.partial_resolve {
                        list.ResolveSubresourceRegion(
                            &resolve.resource,
                            destination_subresource,
                            rect.left as u32,
                            rect.top as u32,
                            &target.resource,
                            source_subresource,
                            Some(rect),
                            target.format,
                            D3D12_RESOLVE_MODE_AVERAGE,
                        );
                    } else {
                        encoder.list.ResolveSubresource(
                            &resolve.resource,
                            destination_subresource,
                            &target.resource,
                            source_subresource,
                            target.format,
                        );
                    }
                }
                super::super::super::transition::barrier_sub(
                    &encoder.list,
                    &target.resource,
                    source_subresource,
                    D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                );
                super::super::super::transition::barrier_sub(
                    &encoder.list,
                    &resolve.resource,
                    destination_subresource,
                    D3D12_RESOURCE_STATE_RESOLVE_DEST,
                    destination_before,
                );
            }
        } else if target.surface {
            barrier(
                &encoder.list,
                &target.resource,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PRESENT,
            );
        }
    }
    if let Some((set, slot)) = state.end_timestamp {
        super::super::super::query::timestamp(encoder, &set, slot)?;
    }
    Ok(())
}
