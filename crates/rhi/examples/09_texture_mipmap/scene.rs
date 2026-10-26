//! Mipmap generation example workload.
#[path = "pipeline.rs"]
mod pipeline;
#[path = "recording.rs"]
mod recording;
#[path = "resources.rs"]
mod resources;

use fluxel_rhi::api::*;
use fluxel_rhi::backend::*;
pub use pipeline::Tunnel;
pub use recording::FrameCommit;
pub use resources::MipChain;

pub trait MipmapWorkloadBackend:
    Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + CommandBackend
{
}
impl<T> MipmapWorkloadBackend for T where
    T: Backend + ResourceBackend + ShaderBackend + PipelineBackend + RasterBackend + CommandBackend
{
}

pub trait MipmapBackend: MipmapWorkloadBackend + DeferredBackend {}
impl<T> MipmapBackend for T where T: MipmapWorkloadBackend + DeferredBackend {}
