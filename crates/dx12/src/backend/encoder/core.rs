use super::*;
use std::sync::{Arc, Mutex};

/// Direct command objects that can be reset after their submission fence has
/// completed. They never enter this pool while `Dx12RecordedWork` still owns
/// them in the device's in-flight list.
pub(crate) struct DirectCommandPool {
    available: Mutex<Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>>,
}

impl DirectCommandPool {
    const MAX_CACHED: usize = 32;

    pub(crate) fn acquire(
        &self,
        device: &ID3D12Device,
    ) -> Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList), Error> {
        if let Some((allocator, list)) = self
            .available
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop()
        {
            unsafe {
                allocator.Reset().map_err(native_error)?;
                list.Reset(&allocator, None).map_err(native_error)?;
            }
            return Ok((allocator, list));
        }

        unsafe {
            let allocator = device
                .CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .map_err(native_error)?;
            let list = device
                .CreateCommandList::<_, _, ID3D12GraphicsCommandList>(
                    0,
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    &allocator,
                    None,
                )
                .map_err(native_error)?;
            Ok((allocator, list))
        }
    }

    fn release(&self, allocator: ID3D12CommandAllocator, list: ID3D12GraphicsCommandList) {
        let mut available = self
            .available
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if available.len() < Self::MAX_CACHED {
            available.push((allocator, list));
        }
    }
}

impl Default for DirectCommandPool {
    fn default() -> Self {
        Self {
            available: Mutex::new(Vec::new()),
        }
    }
}
/// until this value crosses the queue boundary.
pub struct Dx12Encoder {
    pub(crate) allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    direct_commands: Arc<DirectCommandPool>,
    pub(crate) mip_blit_cache: Arc<super::mipmap::MipBlitCache>,
    pub(crate) uploads: Vec<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    pub(crate) mip_blit_heaps: Vec<ID3D12DescriptorHeap>,
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
        let (allocator, list) = device.0.direct_commands.acquire(&device.0.native)?;
        Ok(Self {
            allocator,
            list,
            direct_commands: device.0.direct_commands.clone(),
            mip_blit_cache: device.0.mip_blit_cache.clone(),
            uploads: Vec::new(),
            mip_blit_heaps: Vec::new(),
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

impl Drop for Dx12Encoder {
    fn drop(&mut self) {
        // `Dx12RecordedWork` remains in the device's in-flight list until the
        // submission fence completes, so a submitted encoder only reaches
        // this pool after the GPU has stopped using its native objects. An
        // unfinished list is still open and cannot legally be reset, so a
        // failed recording simply releases its native objects instead.
        if self.closed {
            self.direct_commands
                .release(self.allocator.clone(), self.list.clone());
        }
    }
}
