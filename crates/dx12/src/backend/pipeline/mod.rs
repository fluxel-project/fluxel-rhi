//! DX12 graphics-pipeline lowering.
//!
//! The implementation is split by native responsibility: root signatures,
//! fixed-function graphics PSOs, and the D3D12 stream for view instancing and
//! per-face stencil masks. The
//! modules create immutable native objects only; encoder modules bind them and
//! submission keeps them alive. A bounded worker handles cold creation only;
//! it never records commands or schedules GPU execution.
pub(super) mod cache;
pub(super) mod fixed;
mod layout;
mod raster;
mod stream;
pub(crate) mod worker;

pub(crate) use layout::{
    IMMEDIATE_REGISTER, IMMEDIATE_SPACE, MAX_IMMEDIATE_BYTES, PipelineLayoutState,
    PushConstantsRoot, create_pipeline_layout, pipeline_layout_support,
};
pub(crate) use raster::RasterPipelineState;
pub(crate) use stream::{create_raster_pso_stream, view_instancing_desc, view_locations};

pub(super) const ROOT_DESCRIPTOR_DWORDS: u32 = 2;
pub(super) const MAX_DYNAMIC_UNIFORM_BUFFERS: u32 =
    windows::Win32::Graphics::Direct3D12::D3D12_MAX_ROOT_COST / ROOT_DESCRIPTOR_DWORDS;
