//! Mipmap generation example workload.
mod pipeline;
mod recording;
mod resources;

use fluxel_rhi::api::*;
pub use pipeline::Tunnel;
pub use resources::MipChain;

pub trait MipmapBackend:
    Backend
    + ResourceBackend
    + ShaderBackend
    + PipelineBackend
    + RasterBackend
    + CommandBackend
    + DeferredBackend
{
}
impl<T> MipmapBackend for T where
    T: Backend
        + ResourceBackend
        + ShaderBackend
        + PipelineBackend
        + RasterBackend
        + CommandBackend
        + DeferredBackend
{
}
