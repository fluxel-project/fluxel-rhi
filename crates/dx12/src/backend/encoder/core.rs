use super::*;
/// until this value crosses the queue boundary.
pub struct Dx12Encoder {
    pub(crate) allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    pub(crate) uploads: Vec<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    pub(crate) mip_blit_heaps: Vec<ID3D12DescriptorHeap>,
    pub(crate) mip_blit_states: Vec<ID3D12PipelineState>,
    pub(crate) mip_blit_roots: Vec<ID3D12RootSignature>,
    pub(crate) command_signatures: Vec<ID3D12CommandSignature>,
    pub(crate) secondary: Vec<super::secondary::Dx12SecondaryWork>,
    pub(crate) queries: Vec<Dx12QuerySet>,
    pub(crate) active_occlusion: Option<(Dx12QuerySet, u32)>,
    // Retain typed handles as well as native COM objects.  In particular,
    // descriptor suballocations are released with the Dx12BindGroup handle.
    pub(crate) bound_groups: Vec<Dx12BindGroup>,
    pub(crate) buffers: Vec<Dx12Buffer>,
    pub(crate) textures: Vec<Dx12Texture>,
    pub(crate) pipeline: Option<Dx12RasterPipeline>,
    pub(crate) compute_pipeline: Option<Dx12ComputePipeline>,
    pub(crate) surface_target: Option<ID3D12Resource>,
    pub(crate) texture_target: Option<ID3D12Resource>,
    pub(crate) depth_target: Option<ID3D12Resource>,
    pub(crate) resolve_target: Option<ID3D12Resource>,
    pub(crate) resolve_format: Option<DXGI_FORMAT>,
    pub(crate) raster_format: Option<api::TextureFormat>,
    pub(crate) raster_samples: Option<u32>,
    pub(crate) raster_multiview: Option<u32>,
    pub(crate) closed: bool,
}

/// A one-shot, closed command list and all resources whose CPU ownership must
/// outlive `ExecuteCommandLists`.
pub struct Dx12RecordedWork {
    pub(crate) lane: LaneId,
    pub(crate) encoder: Dx12Encoder,
}

#[derive(Clone)]
pub struct Dx12Completion {
    pub(crate) fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    pub(crate) value: u64,
}

pub struct RasterExtras;

impl api::RasterExtrasOps<Dx12> for RasterExtras {
    const AVAILABLE: bool = true;
    fn draw_indirect(
        encoder: &mut Dx12Encoder,
        args: &Dx12Buffer,
        offset: u64,
        indexed: bool,
    ) -> Result<(), Error> {
        super::indirect::draw(encoder, args, offset, indexed)
    }
    fn multi_draw(
        encoder: &mut Dx12Encoder,
        args: &Dx12Buffer,
        offset: u64,
        count: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<(), Error> {
        super::indirect::multi(encoder, args, offset, count, stride, indexed)
    }
    fn multi_draw_count(
        _: &mut Dx12Encoder,
        _: &Dx12Buffer,
        _: u64,
        _: &Dx12Buffer,
        _: u64,
        _: u32,
        _: u32,
        _: bool,
    ) -> Result<(), Error> {
        Err(Error::Unsupported("DX12 multi draw count"))
    }
    fn set_push_constants(
        encoder: &mut Dx12Encoder,
        stages: api::StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<(), Error> {
        let pipeline = encoder.pipeline.as_ref().ok_or(Error::InvalidState)?;
        let range = pipeline
            .0
            .layout
            .push_constants
            .as_ref()
            .ok_or(Error::Unsupported("DX12 pipeline has no push constants"))?;
        let end = offset
            .checked_add(u32::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?)
            .ok_or(Error::InvalidRange)?;
        if stages != range.stages
            || bytes.is_empty()
            || offset < range.range.start
            || end > range.range.end
            || offset % 4 != 0
            || bytes.len() % 4 != 0
        {
            return Err(Error::InvalidRange);
        }
        let values: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("exact DWORD")))
            .collect();
        unsafe {
            encoder.list.SetGraphicsRoot32BitConstants(
                range.root,
                values.len() as u32,
                values.as_ptr().cast(),
                (offset - range.range.start) / 4,
            );
        }
        Ok(())
    }
}

impl Dx12Encoder {
    pub(crate) fn new(device: &Dx12Device) -> Result<Self, Error> {
        let (allocator, list) = unsafe {
            let allocator = device
                .0
                .native
                .CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(native_error)?;
            let list = device
                .0
                .native
                .CreateCommandList::<_, _, ID3D12GraphicsCommandList>(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocator,
                    None,
                )
                .map_err(native_error)?;
            (allocator, list)
        };
        Ok(Self {
            allocator,
            list,
            uploads: Vec::new(),
            mip_blit_heaps: Vec::new(),
            mip_blit_states: Vec::new(),
            mip_blit_roots: Vec::new(),
            command_signatures: Vec::new(),
            secondary: Vec::new(),
            queries: Vec::new(),
            active_occlusion: None,
            bound_groups: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            pipeline: None,
            compute_pipeline: None,
            surface_target: None,
            texture_target: None,
            depth_target: None,
            resolve_target: None,
            resolve_format: None,
            raster_format: None,
            raster_samples: None,
            raster_multiview: None,
            closed: false,
        })
    }
}
