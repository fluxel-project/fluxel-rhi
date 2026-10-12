//! DX12 graphics-pipeline lowering.
//!
//! The implementation is split by native responsibility: root signatures,
//! fixed-function graphics PSOs, and the D3D12 view-instancing stream.
mod layout;
mod raster;
mod stream;

pub(crate) use layout::{PipelineLayoutState, create_pipeline_layout};
pub(crate) use raster::RasterPipelineState;
pub(crate) use stream::create_view_instanced_pso;
