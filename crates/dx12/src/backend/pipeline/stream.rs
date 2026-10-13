//! D3D12 pipeline-state stream lowering for view instancing.
//!
//! D3D12 exposes view instancing only through the byte-addressed pipeline
//! stream ABI. The builder below serializes supported POD subobjects in the
//! required alignment order and calls Device2 synchronously; it is intentionally
//! narrow and is not a general command or PSO cache.
use fluxel_rhi_core::api::{Error, Result};
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};
/// Builds the extended PSO stream required by D3D12 view instancing. The
/// legacy graphics-PSO descriptor has no `ViewInstancingDesc` field.
pub(crate) fn create_view_instanced_pso(
    device: &ID3D12Device,
    pso: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    root: &ID3D12RootSignature,
    locations: &[D3D12_VIEW_INSTANCE_LOCATION; 2],
) -> Result<ID3D12PipelineState> {
    fn push<T: Copy>(bytes: &mut Vec<u8>, tag: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, value: T) {
        bytes.resize(bytes.len().next_multiple_of(8), 0);
        bytes.extend_from_slice(&(tag.0 as u32).to_ne_bytes());
        bytes.resize(bytes.len().next_multiple_of(std::mem::align_of::<T>()), 0);
        let value = std::mem::ManuallyDrop::new(value);
        // SAFETY: T is Copy and is held in ManuallyDrop, so reading exactly its
        // initialized object representation neither moves nor drops it. D3D12
        // defines these stream subobjects as their in-memory ABI representation.
        let raw = unsafe {
            std::slice::from_raw_parts((&*value as *const T).cast::<u8>(), std::mem::size_of::<T>())
        };
        bytes.extend_from_slice(raw);
    }

    let mut bytes = Vec::with_capacity(512);
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
        root.as_raw(),
    );
    push(&mut bytes, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS, pso.VS);
    push(&mut bytes, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, pso.PS);
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND,
        pso.BlendState,
    );
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK,
        pso.SampleMask,
    );
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
        pso.RasterizerState,
    );
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
        pso.DepthStencilState,
    );
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
        pso.PrimitiveTopologyType,
    );
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
        D3D12_RT_FORMAT_ARRAY {
            RTFormats: pso.RTVFormats,
            NumRenderTargets: pso.NumRenderTargets,
        },
    );
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
        pso.DSVFormat,
    );
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
        pso.SampleDesc,
    );
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT,
        pso.InputLayout,
    );
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_FLAGS,
        pso.Flags,
    );
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING,
        D3D12_VIEW_INSTANCING_DESC {
            ViewInstanceCount: 2,
            pViewInstanceLocations: locations.as_ptr(),
            Flags: D3D12_VIEW_INSTANCING_FLAG_ENABLE_VIEW_INSTANCE_MASKING,
        },
    );
    let stream = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: bytes.len(),
        pPipelineStateSubobjectStream: bytes.as_mut_ptr().cast(),
    };
    let device: ID3D12Device2 = device.cast().map_err(native_error)?;
    // SAFETY: `bytes` owns the complete aligned stream, including all pointer
    // targets borrowed from `pso`/`locations`, until this synchronous call
    // returns. Device2 copies/consumes the stream before returning.
    unsafe { device.CreatePipelineState::<ID3D12PipelineState>(&stream) }.map_err(native_error)
}

fn native_error(e: windows::core::Error) -> Error {
    Error::Backend(format!("Direct3D 12: {e}"))
}
