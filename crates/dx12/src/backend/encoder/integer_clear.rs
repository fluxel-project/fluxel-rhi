//! Exact integer render-target clears using a private fullscreen draw.
//!
//! D3D12's float clear cannot represent every 32-bit integer. Root constants
//! carry the original bits to a signed or unsigned pixel shader instead. The
//! caller supplies one single-layer RTV, retains it and the returned pipeline
//! through completion, and restores the graphics state disturbed by `record`.
use super::*;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
use windows::Win32::Foundation::{FALSE, TRUE};

pub(crate) struct IntegerClearPipeline {
    root: ID3D12RootSignature,
    state: ID3D12PipelineState,
}

pub(crate) fn volume_slice_rtv(
    device: &Dx12Device,
    texture: &super::super::texture::TextureState,
    mip: u32,
    slice: u32,
) -> Result<(ID3D12DescriptorHeap, D3D12_CPU_DESCRIPTOR_HANDLE), Error> {
    let desc = D3D12_DESCRIPTOR_HEAP_DESC {
        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
        NumDescriptors: 1,
        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
        NodeMask: 0,
    };
    // SAFETY: the heap description is fully initialized and retained by the encoder.
    let heap = unsafe {
        device
            .0
            .native
            .CreateDescriptorHeap::<ID3D12DescriptorHeap>(&desc)
    }
    .map_err(native_error)?;
    let handle = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
    let view = D3D12_RENDER_TARGET_VIEW_DESC {
        Format: super::super::texture::dxgi_format(texture.desc.format),
        ViewDimension: D3D12_RTV_DIMENSION_TEXTURE3D,
        Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
            Texture3D: D3D12_TEX3D_RTV {
                MipSlice: mip,
                FirstWSlice: slice,
                WSize: 1,
            },
        },
    };
    // SAFETY: the caller validates mip and W slice against the retained texture.
    unsafe {
        device
            .0
            .native
            .CreateRenderTargetView(&texture.resource, Some(&view), handle)
    };
    Ok((heap, handle))
}

#[derive(Default)]
pub(crate) struct IntegerClearCache(
    Mutex<HashMap<(api::TextureFormat, u32, bool), Arc<IntegerClearPipeline>>>,
);

impl IntegerClearCache {
    pub(crate) fn get_or_create(
        &self,
        device: &Dx12Device,
        format: api::TextureFormat,
        samples: u32,
        signed: bool,
    ) -> Result<Arc<IntegerClearPipeline>, Error> {
        let mut cache = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pipeline) = cache.get(&(format, samples, signed)) {
            return Ok(pipeline.clone());
        }
        let pipeline = prepare(device, format, samples, signed)?;
        cache.insert((format, samples, signed), pipeline.clone());
        Ok(pipeline)
    }
}

pub(crate) fn prepare(
    device: &Dx12Device,
    format: api::TextureFormat,
    samples: u32,
    signed: bool,
) -> Result<Arc<IntegerClearPipeline>, Error> {
    use api::TextureFormat::*;
    let compatible = if signed {
        matches!(
            format,
            R8Sint
                | Rg8Sint
                | Rgba8Sint
                | R16Sint
                | Rg16Sint
                | Rgba16Sint
                | R32Sint
                | Rg32Sint
                | Rgba32Sint
        )
    } else {
        matches!(
            format,
            R8Uint
                | Rg8Uint
                | Rgba8Uint
                | R16Uint
                | Rg16Uint
                | Rgba16Uint
                | R32Uint
                | Rg32Uint
                | Rgba32Uint
                | Rgb10a2Uint
        )
    };
    if !compatible || !matches!(samples, 1 | 2 | 4 | 8 | 16 | 32) {
        return Err(Error::InvalidState);
    }
    let param = D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Constants: D3D12_ROOT_CONSTANTS {
                ShaderRegister: 0,
                RegisterSpace: 0,
                Num32BitValues: 4,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    };
    let root_desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: 1,
        pParameters: &param,
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let mut blob = None;
    // SAFETY: all root descriptor pointers remain live through serialization.
    unsafe {
        D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)
    }
    .map_err(native_error)?;
    let blob = blob.ok_or(Error::OutOfMemory)?;
    // SAFETY: the serialized blob owns the returned range until creation completes.
    let bytes =
        unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) };
    let root = unsafe {
        device
            .0
            .native
            .CreateRootSignature::<ID3D12RootSignature>(0, bytes)
    }
    .map_err(native_error)?;
    let vs = include_bytes!("../../assets/integer_clear.vs.dxil");
    let ps: &[u8] = if signed {
        include_bytes!("../../assets/integer_clear.sint.dxil")
    } else {
        include_bytes!("../../assets/integer_clear.uint.dxil")
    };
    let blend = D3D12_RENDER_TARGET_BLEND_DESC {
        BlendEnable: FALSE,
        LogicOpEnable: FALSE,
        SrcBlend: D3D12_BLEND_ONE,
        DestBlend: D3D12_BLEND_ZERO,
        BlendOp: D3D12_BLEND_OP_ADD,
        SrcBlendAlpha: D3D12_BLEND_ONE,
        DestBlendAlpha: D3D12_BLEND_ZERO,
        BlendOpAlpha: D3D12_BLEND_OP_ADD,
        LogicOp: D3D12_LOGIC_OP_NOOP,
        RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
    };
    let mut formats = [windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN; 8];
    formats[0] = super::super::texture::dxgi_format(format);
    let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vs.as_ptr().cast(),
            BytecodeLength: vs.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: ps.as_ptr().cast(),
            BytecodeLength: ps.len(),
        },
        BlendState: D3D12_BLEND_DESC {
            AlphaToCoverageEnable: FALSE,
            IndependentBlendEnable: FALSE,
            RenderTarget: [blend; 8],
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            FrontCounterClockwise: FALSE,
            DepthBias: 0,
            DepthBiasClamp: 0.,
            SlopeScaledDepthBias: 0.,
            DepthClipEnable: TRUE,
            MultisampleEnable: if samples > 1 { TRUE } else { FALSE },
            AntialiasedLineEnable: FALSE,
            ForcedSampleCount: 0,
            ConservativeRaster: D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
        },
        DepthStencilState: Default::default(),
        InputLayout: Default::default(),
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: formats,
        DSVFormat: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: samples,
            Quality: 0,
        },
        ..Default::default()
    };
    // SAFETY: PSO references live shader bytes and a live root through this synchronous call.
    let state = unsafe {
        device
            .0
            .native
            .CreateGraphicsPipelineState::<ID3D12PipelineState>(&desc)
    };
    // SAFETY: release the one clone held in the ABI descriptor exactly once.
    unsafe { ManuallyDrop::drop(&mut desc.pRootSignature) };
    Ok(Arc::new(IntegerClearPipeline {
        root,
        state: state.map_err(native_error)?,
    }))
}

pub(crate) fn record(
    list: &ID3D12GraphicsCommandList,
    pipeline: &IntegerClearPipeline,
    rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
    width: u32,
    height: u32,
    rect: RECT,
    values: [u32; 4],
) -> Result<(), Error> {
    if width == 0
        || height == 0
        || rect.left < 0
        || rect.top < 0
        || rect.right <= rect.left
        || rect.bottom <= rect.top
        || rect.right as u32 > width
        || rect.bottom as u32 > height
    {
        return Err(Error::InvalidRange);
    }
    // SAFETY: caller owns the open direct list and retains pipeline and RTV through GPU completion.
    unsafe {
        list.OMSetRenderTargets(1, Some(&rtv), false, None);
        list.SetGraphicsRootSignature(&pipeline.root);
        list.SetPipelineState(&pipeline.state);
        list.SetGraphicsRoot32BitConstants(0, 4, values.as_ptr().cast(), 0);
        list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        list.RSSetViewports(&[D3D12_VIEWPORT {
            TopLeftX: 0.,
            TopLeftY: 0.,
            Width: width as f32,
            Height: height as f32,
            MinDepth: 0.,
            MaxDepth: 1.,
        }]);
        list.RSSetScissorRects(&[rect]);
        list.DrawInstanced(3, 1, 0, 0);
    }
    Ok(())
}
