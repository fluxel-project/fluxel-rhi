//! DX12 graphics-pipeline lowering.
//!
//! The implementation is split by native responsibility: root signatures,
//! fixed-function graphics PSOs, and the D3D12 view-instancing stream. The
//! modules create immutable native objects only; encoder modules bind them and
//! submission keeps them alive. A bounded worker handles cold creation only;
//! it never records commands or schedules GPU execution.
pub(super) mod cache;
pub(super) mod fixed;
mod layout;
mod raster;
mod stream;
pub(crate) mod worker;

pub(crate) use layout::{PipelineLayoutState, create_pipeline_layout};
pub(crate) use raster::RasterPipelineState;
pub(crate) use stream::create_view_instanced_pso;

pub(super) const ROOT_DESCRIPTOR_DWORDS: u32 = 2;
pub(super) const MAX_DYNAMIC_UNIFORM_BUFFERS: u32 =
    windows::Win32::Graphics::Direct3D12::D3D12_MAX_ROOT_COST / ROOT_DESCRIPTOR_DWORDS;
