//! D3D12 pipeline-state stream lowering for extended raster subobjects.
//!
//! D3D12 exposes view instancing only through the byte-addressed pipeline
//! stream ABI. The builder below serializes supported POD subobjects in the
//! required alignment order and calls Device2 synchronously; it is intentionally
//! narrow and is not a general command or PSO cache.
use fluxel_rhi_core::api::{Error, Result};
use std::mem::{MaybeUninit, align_of, size_of};
use windows::{Win32::Graphics::Direct3D12::*, core::Interface};

pub(crate) fn create_raster_pso_stream(
    device: &ID3D12Device,
    pso: &D3D12_GRAPHICS_PIPELINE_STATE_DESC,
    root: &ID3D12RootSignature,
    locations: Option<&[D3D12_VIEW_INSTANCE_LOCATION]>,
    depth_stencil: Option<D3D12_DEPTH_STENCIL_DESC2>,
    cached: Option<&[u8]>,
) -> Result<ID3D12PipelineState> {
    fn push<T: Copy>(
        words: &mut Vec<MaybeUninit<u64>>,
        tag: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE,
        value: T,
    ) {
        let alignment = align_of::<T>();
        assert!(alignment <= align_of::<u64>());
        let offset = size_of::<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE>().next_multiple_of(alignment);
        let start = words.len();
        words.resize_with(
            start + (offset + size_of::<T>()).div_ceil(size_of::<u64>()),
            MaybeUninit::zeroed,
        );
        let record = words[start..].as_mut_ptr().cast::<u8>();
        // SAFETY: the allocated record is 8-byte aligned and contains aligned
        // storage for the tag and payload. Payload padding is never read in Rust.
        unsafe {
            record
                .cast::<D3D12_PIPELINE_STATE_SUBOBJECT_TYPE>()
                .write(tag);
            record.add(offset).cast::<T>().write(value);
        }
    }

    let mut bytes = Vec::with_capacity(64);
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
    if let Some(depth) = depth_stencil {
        push(
            &mut bytes,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL2,
            depth,
        );
    } else {
        push(
            &mut bytes,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
            pso.DepthStencilState,
        );
    }
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY,
        pso.PrimitiveTopologyType,
    );
    push(
        &mut bytes,
        D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_IB_STRIP_CUT_VALUE,
        pso.IBStripCutValue,
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
    if let Some(locations) = locations {
        push(
            &mut bytes,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING,
            D3D12_VIEW_INSTANCING_DESC {
                ViewInstanceCount: locations.len() as u32,
                pViewInstanceLocations: locations.as_ptr(),
                Flags: D3D12_VIEW_INSTANCING_FLAG_ENABLE_VIEW_INSTANCE_MASKING,
            },
        );
    }
    if let Some(blob) = cached {
        push(
            &mut bytes,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO,
            D3D12_CACHED_PIPELINE_STATE {
                pCachedBlob: blob.as_ptr().cast(),
                CachedBlobSizeInBytes: blob.len(),
            },
        );
    }
    let stream = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: std::mem::size_of_val(bytes.as_slice()),
        pPipelineStateSubobjectStream: bytes.as_mut_ptr().cast(),
    };
    let device: ID3D12Device2 = device.cast().map_err(native_error)?;
    // SAFETY: `bytes` owns the complete aligned stream, including all pointer
    // targets borrowed from `pso`/`locations`, until this synchronous call
    // returns. Device2 copies/consumes the stream before returning.
    match unsafe { device.CreatePipelineState::<ID3D12PipelineState>(&stream) } {
        Err(error) if cached.is_some() && super::cache::is_compatibility_error(&error) => {
            create_raster_pso_stream(
                &device.cast().map_err(native_error)?,
                pso,
                root,
                locations,
                depth_stencil,
                None,
            )
        }
        result => result.map_err(native_error),
    }
}

fn native_error(e: windows::core::Error) -> Error {
    crate::backend::native_error(e)
}
