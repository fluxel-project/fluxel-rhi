//! Select depth descriptors for multiview load clears.
use super::*;

pub(super) fn selected(mask: Option<u32>, layer: u32) -> bool {
    mask.map_or(true, |mask| {
        1u32.checked_shl(layer).is_some_and(|bit| mask & bit != 0)
    })
}

pub(super) fn depth_clear_handles(
    encoder: &mut Dx12Encoder,
    desc: &api::RasterPassDesc<'_, Dx12>,
    dsv: Option<D3D12_CPU_DESCRIPTOR_HANDLE>,
    depth_clear: Option<f32>,
    stencil_clear: Option<u8>,
) -> Result<Vec<D3D12_CPU_DESCRIPTOR_HANDLE>, Error> {
    let mut depth_clear_handles = Vec::new();
    if (depth_clear.is_some() || stencil_clear.is_some()) && desc.multiview_mask.is_some() {
        let depth = desc.depth_stencil.as_ref().ok_or(Error::InvalidState)?;
        let api::AttachmentView::Texture(view) = depth.view else {
            return Err(Error::InvalidState);
        };
        let n = unsafe { view.native_ref() };
        let texture = Dx12Texture(n.0.texture.clone());
        for layer in n.0.desc.layers.clone() {
            if !selected(desc.multiview_mask, layer - n.0.desc.layers.start) {
                continue;
            }
            let mut vd = n.0.desc.clone();
            vd.layers = layer..layer + 1;
            let v = crate::backend::texture::create_view(&encoder.device, &texture, &vd)?;
            let has_stencil = matches!(
                n.0.desc.format,
                api::TextureFormat::Depth24PlusStencil8 | api::TextureFormat::Depth32FloatStencil8
            );
            let handle = match (
                depth.depth_read_only,
                depth.stencil_read_only && has_stencil,
            ) {
                (false, false) => v.0.dsv,
                (true, false) => v.0.dsv_read_only_depth,
                (false, true) => v.0.dsv_read_only_stencil,
                (true, true) => v.0.dsv_read_only_both,
            }
            .ok_or(Error::InvalidState)?;
            encoder
                .clear_heaps
                .push(v.0.dsv_heap.as_ref().ok_or(Error::InvalidState)?.clone());
            depth_clear_handles.push(handle);
        }
    } else if let Some(handle) = dsv {
        depth_clear_handles.push(handle);
    }
    Ok(depth_clear_handles)
}

pub(super) fn color_clear_handles(
    encoder: &mut Dx12Encoder,
    mask: Option<u32>,
    view: &api::AttachmentView<'_, Dx12>,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    float_clear: Option<[f32; 4]>,
    clears: &mut Vec<(D3D12_CPU_DESCRIPTOR_HANDLE, [f32; 4])>,
) -> Result<(), Error> {
    if let Some(c) = float_clear {
        if let (Some(mask), api::AttachmentView::Texture(view)) = (mask, view) {
            let n = unsafe { view.native_ref() };
            let texture = Dx12Texture(n.0.texture.clone());
            for layer in n.0.desc.layers.clone() {
                if !selected(Some(mask), layer - n.0.desc.layers.start) {
                    continue;
                }
                let mut vd = n.0.desc.clone();
                vd.layers = layer..layer + 1;
                let v = crate::backend::texture::create_view(&encoder.device, &texture, &vd)?;
                encoder
                    .clear_heaps
                    .push(v.0.rtv_heap.as_ref().ok_or(Error::InvalidState)?.clone());
                clears.push((v.0.rtv.ok_or(Error::InvalidState)?, c));
            }
        } else {
            clears.push((rtv, c));
        }
    }
    Ok(())
}
