//! DX12 graphics-pipeline lowering.
//!
//! The implementation is split by native responsibility: root signatures,
//! fixed-function graphics PSOs, and the D3D12 view-instancing stream. The
//! modules create immutable native objects only; encoder modules bind them and
//! submission keeps them alive. No pipeline cache or compilation scheduler is
//! hidden behind this boundary.
mod layout;
mod raster;
mod stream;

pub(crate) use layout::{PipelineLayoutState, create_pipeline_layout};
pub(crate) use raster::RasterPipelineState;
pub(crate) use stream::create_view_instanced_pso;
