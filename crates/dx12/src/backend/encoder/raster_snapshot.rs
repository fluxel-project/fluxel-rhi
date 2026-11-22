//! Current primary graphics state needed after executing a secondary bundle.
use super::*;
use std::collections::BTreeMap;

#[derive(Clone, Copy)]
pub(crate) enum RootValue {
    Table(D3D12_GPU_DESCRIPTOR_HANDLE),
    Cbv(u64),
}
#[derive(Default)]
pub(crate) struct RasterSnapshot {
    pub roots: BTreeMap<u32, RootValue>,
    pub constants: BTreeMap<(u32, u32), u32>,
    pub vertices: BTreeMap<u32, D3D12_VERTEX_BUFFER_VIEW>,
    pub index: Option<D3D12_INDEX_BUFFER_VIEW>,
    pub viewport: Option<D3D12_VIEWPORT>,
    pub scissor: Option<RECT>,
    pub blend: Option<[f32; 4]>,
    pub stencil: [u32; 2],
    pub depth_bias: Option<api::DynamicDepthBias>,
}
impl RasterSnapshot {
    pub fn clear_roots(&mut self) {
        self.roots.clear();
        self.constants.clear();
    }
}

pub(crate) fn restore(encoder: &mut Dx12Encoder) -> Result<(), Error> {
    let s = &encoder.raster_snapshot;
    // All referenced resources, descriptors, and PSOs are retained by the encoder.
    unsafe {
        // Invalidate arguments left by the bundle even when both pipelines
        // use the same root signature, then replay only primary-owned values.
        encoder
            .list
            .SetGraphicsRootSignature(None::<&ID3D12RootSignature>);
        if let Some(p) = &encoder.pipeline {
            encoder.list.SetGraphicsRootSignature(&p.0.root);
            encoder.list.SetPipelineState(&p.0.state);
            encoder.list.IASetPrimitiveTopology(p.0.topology);
        } else if let Some(p) = &encoder.mesh_pipeline {
            encoder.list.SetGraphicsRootSignature(&p.0.layout.root);
            encoder.list.SetPipelineState(&p.0.state);
        }
        if let Some(value) = s.depth_bias {
            let list9 = encoder
                .list
                .cast::<ID3D12GraphicsCommandList9>()
                .map_err(native_error)?;
            list9.RSSetDepthBias(value.constant as f32, value.clamp, value.slope_scale);
        }
        for (&root, value) in &s.roots {
            match value {
                RootValue::Table(handle) => {
                    encoder.list.SetGraphicsRootDescriptorTable(root, *handle)
                }
                RootValue::Cbv(address) => encoder
                    .list
                    .SetGraphicsRootConstantBufferView(root, *address),
            }
        }
        for (&(root, offset), &value) in &s.constants {
            encoder
                .list
                .SetGraphicsRoot32BitConstant(root, value, offset);
        }
        // Explicitly clear slots modified by the bundle that the primary never bound.
        let empty = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        encoder.list.IASetVertexBuffers(0, Some(&empty));
        for (&slot, view) in &s.vertices {
            encoder
                .list
                .IASetVertexBuffers(slot, Some(std::slice::from_ref(view)));
        }
        encoder
            .list
            .IASetIndexBuffer(s.index.as_ref().map(|view| view as *const _));
        if let Some(v) = s.viewport {
            encoder.list.RSSetViewports(&[v]);
        }
        if let Some(r) = s.scissor {
            encoder.list.RSSetScissorRects(&[r]);
        }
        encoder
            .list
            .OMSetBlendFactor(Some(&s.blend.unwrap_or([1.0; 4])));
    }
    write_stencil_references(&encoder.list, s.stencil)
}

pub(super) fn write_stencil_references(
    list: &ID3D12GraphicsCommandList,
    [front, back]: [u32; 2],
) -> Result<(), Error> {
    if front == back {
        // SAFETY: the list is a live graphics list and the command records one reference.
        unsafe { list.OMSetStencilRef(front) };
    } else {
        let list = list
            .cast::<ID3D12GraphicsCommandList8>()
            .map_err(|_| Error::Unsupported("DX12 per-face stencil reference entry point"))?;
        // SAFETY: the per-face feature was checked before this state was recorded.
        unsafe { list.OMSetFrontAndBackStencilRef(front, back) };
    }
    Ok(())
}

pub(super) fn write_blend_constant(
    list: &ID3D12GraphicsCommandList,
    value: [f32; 4],
) -> Result<(), Error> {
    if value.iter().any(|component| !component.is_finite()) {
        return Err(Error::InvalidRange);
    }
    // SAFETY: the list is open, and D3D12 copies the four values during the call.
    unsafe { list.OMSetBlendFactor(Some(&value)) };
    Ok(())
}
