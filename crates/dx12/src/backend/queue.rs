//! Compute queue, native timeline, and fence-retired command ownership.
//!
//! The device's submission gate serializes packet handoff and query state.
//! Each native queue has its own lock and timeline; completion from one fence
//! is never evidence for work on another queue.
use super::*;
use std::sync::{Arc, Mutex, atomic::AtomicU64};
use windows::Win32::Graphics::Direct3D12::*;

pub(crate) struct MultiQueueState {
    pub(crate) queue: ID3D12CommandQueue,
    pub(crate) fence: ID3D12Fence,
    pub(crate) next_fence: AtomicU64,
    pub(crate) lock: Mutex<()>,
    pub(crate) in_flight: Mutex<Vec<(u64, Vec<encoder::Dx12RecordedWork>)>>,
    pub(crate) commands: Arc<ComputeCommandPool>,
}
impl MultiQueueState {
    pub(crate) fn new(device: &ID3D12Device) -> Result<Self> {
        let desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        // SAFETY: the descriptor is initialized and returned COM interfaces are owned.
        let queue = unsafe { device.CreateCommandQueue::<ID3D12CommandQueue>(&desc) }
            .map_err(native_error)?;
        // SAFETY: newly created fence has an independent monotonically increasing timeline.
        let fence = unsafe { device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }
            .map_err(native_error)?;
        Ok(Self {
            queue,
            fence,
            next_fence: AtomicU64::new(1),
            lock: Mutex::new(()),
            in_flight: Mutex::new(Vec::new()),
            commands: Arc::default(),
        })
    }
}
#[derive(Default)]
pub(crate) struct ComputeCommandPool {
    available: Mutex<Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)>>,
}
impl ComputeCommandPool {
    pub(crate) fn acquire(
        &self,
        device: &ID3D12Device,
    ) -> Result<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> {
        if let Some((allocator, list)) = self
            .available
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .pop()
        {
            // SAFETY: retirement proves prior compute execution has completed.
            unsafe {
                allocator.Reset().map_err(native_error)?;
                list.Reset(&allocator, None).map_err(native_error)?;
            }
            return Ok((allocator, list));
        }
        // SAFETY: allocator and list are created together for the compute queue type.
        unsafe {
            let allocator = device
                .CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_COMPUTE)
                .map_err(native_error)?;
            let list = device
                .CreateCommandList::<_, _, ID3D12GraphicsCommandList>(
                    0,
                    D3D12_COMMAND_LIST_TYPE_COMPUTE,
                    &allocator,
                    None,
                )
                .map_err(native_error)?;
            Ok((allocator, list))
        }
    }
    pub(crate) fn release(
        &self,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
    ) {
        let mut available = self.available.lock().unwrap_or_else(|p| p.into_inner());
        if available.len() < 32 {
            available.push((allocator, list));
        }
    }
}

/// Queue signal and wait are native GPU operations. The caller holds the
/// submission gate across multi-operation handoff to preserve packet order.
pub(crate) fn signal(device: &Dx12Device, ordinal: u32) -> Result<u64> {
    let (queue, fence, counter, lock) = match ordinal {
        0 => (
            &device.0.queue,
            &device.0.fence,
            &device.0.next_fence,
            &device.0.queue_lock,
        ),
        1 => (
            &device.0.multi_queue.queue,
            &device.0.multi_queue.fence,
            &device.0.multi_queue.next_fence,
            &device.0.multi_queue.lock,
        ),
        _ => return Err(Error::WrongLane),
    };
    let _guard = lock.lock().unwrap_or_else(|p| p.into_inner());
    let value = counter
        .fetch_update(
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
            |v| v.checked_add(1),
        )
        .map_err(|_| Error::InvalidRange)?;
    // SAFETY: queue access is serialized and device retains queue/fence.
    unsafe { queue.Signal(fence, value) }.map_err(native_error)?;
    Ok(value)
}
pub(crate) fn wait(
    device: &Dx12Device,
    ordinal: u32,
    fence: &ID3D12Fence,
    value: u64,
) -> Result<()> {
    let (queue, lock) = match ordinal {
        0 => (&device.0.queue, &device.0.queue_lock),
        1 => (&device.0.multi_queue.queue, &device.0.multi_queue.lock),
        _ => return Err(Error::WrongLane),
    };
    let _guard = lock.lock().unwrap_or_else(|p| p.into_inner());
    // SAFETY: queue access is serialized and fence remains alive on the device
    // or the owned signal token through this native call.
    unsafe { queue.Wait(fence, value) }.map_err(native_error)
}
pub(crate) fn fence(device: &Dx12Device, ordinal: u32) -> Result<&ID3D12Fence> {
    match ordinal {
        0 => Ok(&device.0.fence),
        1 => Ok(&device.0.multi_queue.fence),
        _ => Err(Error::WrongLane),
    }
}
