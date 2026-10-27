//! Explicit, read-only observations of backend-owned resource pools and native
//! memory reports. These facts are separate from [`RhiStatsSnapshot`], which
//! counts accepted Core requests and never queries a driver.

use super::{Device, Result};
use crate::backend::{Backend, MemoryObservationOps};

/// Backend facts returned by one explicit memory observation call.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MemorySnapshot {
    /// Summary of a backend-owned suballocation pool, when that backend tracks
    /// one. This does not include unrelated committed, imported, placed,
    /// sparse, staging, or driver-internal allocations.
    pub allocator: Option<AllocatorMemorySnapshot>,
    /// Native process, segment, or heap reports. Values are estimates or
    /// operating-system reports according to [`MemoryReportSource`]; they are
    /// not per-RHI-pool totals or resident-byte measurements.
    pub native_reports: Vec<NativeMemoryReport>,
}

/// A summary of one backend-owned page/range allocator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocatorMemorySnapshot {
    /// Which backend-owned pool produced this exact range accounting.
    pub source: AllocatorMemorySource,
    /// Total bytes in native backing pages retained by the pool, including
    /// empty cached pages.
    pub reserved_bytes: u64,
    /// Bytes currently leased to resources. This is allocated range size,
    /// including backend alignment padding, not the resource's logical size.
    /// In-flight and quarantined resources remain live until their native
    /// owner releases the lease.
    pub live_range_bytes: u64,
    /// Bytes currently available in free ranges within retained pages.
    pub free_bytes: u64,
    /// Largest retained backing-page total observed since device creation.
    pub peak_reserved_bytes: u64,
    /// Largest live suballocation-range total observed since device creation.
    pub peak_live_range_bytes: u64,
    /// Number of retained backing pages at this observation.
    pub page_count: u32,
    /// Number of retained pages that have no live resource leases.
    pub idle_page_count: u32,
    /// Number of live suballocation leases.
    pub live_allocation_count: u32,
}

/// Source and exact coverage of a pool summary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AllocatorMemorySource {
    /// DX12 `DefaultResourcePool` placed-resource pages. Committed fallback,
    /// explicit heaps, sparse/external resources, staging pools and driver
    /// allocations are excluded.
    Dx12DefaultResourcePool,
    /// Vulkan `DefaultResourcePool` buffer/optimal-image pages. Dedicated
    /// allocations, explicit heaps, sparse/external resources, staging pools
    /// and driver allocations are excluded.
    VulkanDefaultResourcePool,
}

/// Provenance of a native usage/budget report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryReportSource {
    /// DXGI's current process usage and OS-provided budget for one memory
    /// segment group.
    DxgiProcessSegment,
    /// Vulkan's implementation-dependent estimated process usage and budget
    /// for one physical memory heap.
    VulkanProcessHeapEstimate,
}

/// Scope of one native usage/budget report.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryReportScope {
    /// DXGI local segment group.
    DxgiLocalSegment,
    /// DXGI non-local segment group.
    DxgiNonLocalSegment,
    /// Vulkan physical memory heap. `device_local` is copied from the
    /// physical-device heap flags; the report remains an implementation-
    /// dependent estimate, not a residency measurement.
    VulkanHeap {
        heap_index: u32,
        heap_size_bytes: u64,
        device_local: bool,
    },
}

/// One native memory usage/budget report. `None` means that value was
/// unavailable; it is never encoded as zero.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NativeMemoryReport {
    /// Native API and accounting scope.
    pub source: MemoryReportSource,
    /// Segment or physical heap represented by this report.
    pub scope: MemoryReportScope,
    /// Usage bytes reported by the OS or estimated by the Vulkan driver.
    pub usage_bytes: Option<u64>,
    /// Budget bytes reported by the OS or estimated by the Vulkan driver.
    pub budget_bytes: Option<u64>,
}

/// An explicit observation domain, separate from request counters.
pub struct MemoryObservationDomain<'a, B: Backend> {
    device: &'a Device<B>,
}

impl<B: Backend> Device<B> {
    /// Returns this device's optional native memory observation domain.
    ///
    /// A supported domain may perform explicit driver queries when
    /// [`MemoryObservationDomain::snapshot`] is called. Unsupported devices
    /// return `None`; no memory observation is folded into [`Device::stats`].
    pub fn memory_observation(&self) -> Option<MemoryObservationDomain<'_, B>> {
        (self.caps.memory_observation && B::MemoryObservation::AVAILABLE)
            .then_some(MemoryObservationDomain { device: self })
    }
}

impl<B: Backend> MemoryObservationDomain<'_, B> {
    /// Reads the backend pool summary and any available native reports.
    ///
    /// This is deliberately explicit because driver budget queries may be
    /// expensive and return process- or heap-scoped estimates that can change
    /// independently of RHI allocations.
    pub fn snapshot(&self) -> Result<MemorySnapshot> {
        B::MemoryObservation::snapshot(&self.device.native)
    }
}
