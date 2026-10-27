//! Direct command-list allocation and encoder-owned lifetime retention.
//!
//! Each encoder owns one current allocator/list pair and may close earlier
//! pairs at explicit external-synchronization boundaries. Submission keeps all
//! of that native work alive through its fence;
//! only retired work returns the pair to `DirectCommandPool`.  The pool reuses
//! native D3D12 objects but does not retain, reorder, or replay commands.
//! Resource references on `Dx12Encoder` exist solely to keep the native
//! objects and descriptor suballocations valid until GPU completion.

use super::*;
use std::sync::{Arc, Mutex};

/// Direct command objects that can be reset after their submission fence has
/// completed. They never enter this pool while `Dx12RecordedWork` still owns
/// them in the device's in-flight list.
pub(crate) struct DirectCommandPool {
    available: Mutex<Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>>,
}

/// A closed primary command list retained by one encoder after an explicit
/// queue-level external synchronization boundary. It is not a replay record:
/// D3D12 commands have already been recorded into this native list.
pub(crate) struct ClosedPrimarySegment {
    pub(crate) allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
}

/// Native queue operation issued immediately after the matching closed
/// segment. `Wait` and `Signal` are D3D12 queue operations, not command-list
/// commands, so retaining this narrow sequence preserves Core call order.
pub(crate) enum QueueBoundary {
    Wait { fence: ID3D12Fence, value: u64 },
    Signal { fence: ID3D12Fence, value: u64 },
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
            // SAFETY: a pair enters `available` only after the submission
            // fence retired its prior work; the mutex grants exclusive reset.
            unsafe {
                allocator.Reset().map_err(native_error)?;
                list.Reset(&allocator, None).map_err(native_error)?;
            }
            return Ok((allocator, list));
        }

        // SAFETY: this device created both objects. The returned allocator and
        // initially-open list are owned by the newly constructed encoder.
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
    /// Retains the native device for backend-private allocations such as a
    /// readback staging resource. This is ownership only: it does not make
    /// the encoder a resource scheduler or state tracker.
    pub(crate) device: Dx12Device,
    pub(crate) allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Native lists closed before an external queue boundary. Boundary `i`
    /// runs after `closed_segments[i]`; the open `list` is always final.
    pub(crate) closed_segments: Vec<ClosedPrimarySegment>,
    pub(crate) queue_boundaries: Vec<QueueBoundary>,
    direct_commands: Arc<DirectCommandPool>,
    pub(crate) is_compute: bool,
    pub(crate) mip_blit_cache: Arc<super::mipmap::MipBlitCache>,
    pub(crate) uploads: Vec<windows::Win32::Graphics::Direct3D12::ID3D12Resource>,
    pub(crate) upload_arena: super::upload::UploadArena,
    pub(crate) compute_snapshot: super::super::compute::ComputeSnapshot,
    pub(crate) mip_blit_heaps: Vec<ID3D12DescriptorHeap>,
    pub(crate) mip_blit_descriptors: Vec<super::super::descriptor::TemporaryDescriptorPair>,
    pub(crate) clear_heaps: Vec<ID3D12DescriptorHeap>,
    /// Leases for CPU descriptors recorded as raw handles. `Dx12RecordedWork`
    /// retains this encoder through the queue fence (or u64::MAX quarantine).
    pub(crate) view_descriptor_leases: Vec<super::super::descriptor::CpuDescriptorLease>,
    pub(crate) integer_clear_pipelines: Vec<Arc<super::integer_clear::IntegerClearPipeline>>,
    pub(crate) secondary: Vec<super::secondary::Dx12SecondaryWork>,
    pub(crate) queries: Vec<Dx12QuerySet>,
    pub(crate) active_occlusion: Option<(Dx12QuerySet, u32)>,
    pub(crate) active_statistics: Option<(Dx12QuerySet, u32)>,
    pub(crate) query_ops: Vec<super::query::QueryOp>,
    // Retain typed handles as well as native COM objects.  In particular,
    // descriptor suballocations are released with the Dx12BindGroup handle.
    pub(crate) bound_groups: Vec<Dx12BindGroup>,
    pub(crate) raster_bound_groups: Vec<Option<Dx12BindGroup>>,
    pub(crate) compute_bound_groups: Vec<Option<Dx12BindGroup>>,
    pub(crate) dynamic_view_tables: Vec<Arc<super::super::descriptor::DynamicViewTable>>,
    pub(crate) buffers: Vec<Dx12Buffer>,
    pub(crate) textures: Vec<Dx12Texture>,
    pub(crate) accelerations: Vec<super::super::Dx12Acceleration>,
    /// TLAS builds recorded in each native-list segment. Submission publishes
    /// them only after that segment crossed `ExecuteCommandLists`.
    pub(crate) ray_builds: Vec<super::super::ray::RecordedTlasBuild>,
    pub(crate) pipeline: Option<Dx12RasterPipeline>,
    pub(crate) raster_pipelines: Vec<Dx12RasterPipeline>,
    pub(crate) compute_pipelines: Vec<Dx12ComputePipeline>,
    pub(crate) compute_pipeline: Option<Dx12ComputePipeline>,
    pub(crate) ray_pipeline: Option<super::super::Dx12RayPipeline>,
    pub(crate) ray_pipelines: Vec<super::super::Dx12RayPipeline>,
    pub(crate) mesh_pipeline: Option<super::super::Dx12MeshPipeline>,
    pub(crate) mesh_pipelines: Vec<super::super::Dx12MeshPipeline>,
    pub(crate) indirect_signatures: Vec<ID3D12CommandSignature>,
    pub(crate) raster_snapshot: super::raster_snapshot::RasterSnapshot,
    pub(crate) raster_state: Option<super::raster::RasterState>,
    pub(crate) raster_formats: Vec<Option<api::TextureFormat>>,
    pub(crate) raster_index_format: Option<api::IndexFormat>,
    pub(crate) raster_depth_format: Option<api::TextureFormat>,
    pub(crate) raster_samples: Option<u32>,
    pub(crate) raster_multiview: Option<u32>,
    /// Native BeginEvent/EndEvent nesting for the currently open command list.
    /// External queue boundaries close that list, so a group cannot cross one.
    pub(crate) debug_group_depth: u32,
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

impl fluxel_rhi_core::backend::RasterExtrasOps<Dx12> for RasterExtras {
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
        encoder: &mut Dx12Encoder,
        args: &Dx12Buffer,
        offset: u64,
        count: &Dx12Buffer,
        count_offset: u64,
        max: u32,
        stride: u32,
        indexed: bool,
    ) -> Result<(), Error> {
        super::indirect::multi_count(
            encoder,
            args,
            offset,
            count,
            count_offset,
            max,
            stride,
            indexed,
        )
    }
    fn set_push_constants(
        encoder: &mut Dx12Encoder,
        stages: api::StageMask,
        offset: u32,
        bytes: &[u8],
    ) -> Result<(), Error> {
        let layout = encoder
            .pipeline
            .as_ref()
            .map(|p| &p.0.layout)
            .or_else(|| encoder.mesh_pipeline.as_ref().map(|p| &p.0.layout))
            .ok_or(Error::InvalidState)?;
        let range = layout
            .push_constants
            .as_ref()
            .ok_or(Error::Unsupported("DX12 pipeline has no push constants"))?;
        let end = offset
            .checked_add(u32::try_from(bytes.len()).map_err(|_| Error::InvalidRange)?)
            .ok_or(Error::InvalidRange)?;
        if bytes.is_empty() || offset % 4 != 0 || bytes.len() % 4 != 0 {
            return Err(Error::InvalidRange);
        }
        let roots = range.graphics_writes(stages, offset, end)?;
        let values: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().expect("exact DWORD")))
            .collect();
        for root in roots {
            for (i, value) in values.iter().copied().enumerate() {
                encoder
                    .raster_snapshot
                    .constants
                    .insert((root, offset / 4 + i as u32), value);
            }
            // SAFETY: the validated root and range belong to the bound signature; values are live during this call.
            unsafe {
                encoder.list.SetGraphicsRoot32BitConstants(
                    root,
                    values.len() as u32,
                    values.as_ptr().cast(),
                    offset / 4,
                );
            }
        }
        Ok(())
    }
}

impl Dx12Encoder {
    pub(crate) fn new(device: &Dx12Device, lane: LaneId) -> Result<Self, Error> {
        let is_compute = lane.ordinal == 1;
        let (allocator, list) = if is_compute {
            device.0.multi_queue.commands.acquire(&device.0.native)?
        } else {
            device.0.direct_commands.acquire(&device.0.native)?
        };
        super::heaps::install(device, &list);
        Ok(Self {
            device: device.clone(),
            allocator,
            list,
            closed_segments: Vec::new(),
            queue_boundaries: Vec::new(),
            direct_commands: device.0.direct_commands.clone(),
            is_compute,
            mip_blit_cache: device.0.mip_blit_cache.clone(),
            uploads: Vec::new(),
            upload_arena: super::upload::UploadArena::new(device.0.upload_pool.clone()),
            compute_snapshot: Default::default(),
            mip_blit_heaps: Vec::new(),
            mip_blit_descriptors: Vec::new(),
            clear_heaps: Vec::new(),
            view_descriptor_leases: Vec::new(),
            integer_clear_pipelines: Vec::new(),
            secondary: Vec::new(),
            queries: Vec::new(),
            active_occlusion: None,
            active_statistics: None,
            query_ops: Vec::new(),
            bound_groups: Vec::new(),
            raster_bound_groups: Vec::new(),
            compute_bound_groups: Vec::new(),
            dynamic_view_tables: Vec::new(),
            buffers: Vec::new(),
            textures: Vec::new(),
            accelerations: Vec::new(),
            ray_builds: Vec::new(),
            pipeline: None,
            raster_pipelines: Vec::new(),
            compute_pipelines: Vec::new(),
            compute_pipeline: None,
            ray_pipeline: None,
            ray_pipelines: Vec::new(),
            mesh_pipeline: None,
            mesh_pipelines: Vec::new(),
            indirect_signatures: Vec::new(),
            raster_snapshot: Default::default(),
            raster_state: None,
            raster_formats: Vec::new(),
            raster_index_format: None,
            raster_depth_format: None,
            raster_samples: None,
            raster_multiview: None,
            debug_group_depth: 0,
            closed: false,
        })
    }
}

impl Dx12Encoder {
    /// Ends the current native list at an explicit Core external boundary,
    /// retains it for submission, and opens the following list. The operation
    /// records no CPU command stream and never derives a dependency.
    pub(crate) fn split_external_boundary(&mut self, boundary: QueueBoundary) -> Result<(), Error> {
        if self.closed
            || self.active_occlusion.is_some()
            || self.active_statistics.is_some()
            || self.debug_group_depth != 0
        {
            return Err(Error::InvalidState);
        }
        let (next_allocator, next_list) = if self.is_compute {
            self.device
                .0
                .multi_queue
                .commands
                .acquire(&self.device.0.native)?
        } else {
            self.direct_commands.acquire(&self.device.0.native)?
        };
        // SAFETY: this encoder exclusively owns its current still-open list.
        // A failed close leaves the encoder unusable; the caller's Core path
        // poisons it and no closed list is returned to an object pool.
        unsafe { self.list.Close() }.map_err(native_error)?;
        let prior = ClosedPrimarySegment {
            allocator: std::mem::replace(&mut self.allocator, next_allocator),
            list: std::mem::replace(&mut self.list, next_list),
        };
        self.closed_segments.push(prior);
        self.queue_boundaries.push(boundary);
        debug_assert_eq!(self.closed_segments.len(), self.queue_boundaries.len());
        super::heaps::install(&self.device, &self.list);
        // The fresh list has no bound native state. Reinstall only the current
        // native state snapshots; this is state restoration, never command
        // replay. Core prohibits an external boundary inside an active pass.
        if self.pipeline.is_some() || self.mesh_pipeline.is_some() {
            super::raster_snapshot::restore(self)?;
        } else if self.compute_pipeline.is_some() || self.ray_pipeline.is_some() {
            super::super::compute::restore_compute_state(self)?;
        }
        Ok(())
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
            let release = |allocator, list| {
                if self.is_compute {
                    self.device.0.multi_queue.commands.release(allocator, list);
                } else {
                    self.direct_commands.release(allocator, list);
                }
            };
            for segment in self.closed_segments.drain(..) {
                release(segment.allocator, segment.list);
            }
            release(self.allocator.clone(), self.list.clone());
        }
    }
}
