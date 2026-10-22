//! Focused Vulkan pipeline lowering modules.
//!
//! These modules turn already validated Core descriptions into Vulkan layouts,
//! descriptor sets, and pipelines. Their RAII wrappers retain every native
//! dependency until command recording and fence-owned submission retention no
//! longer reference it; they do not schedule work or infer resource hazards.

mod binding;
pub(crate) mod cache;
mod compute;
pub(crate) mod fixed;
mod vertex_format;
pub(crate) use vertex_format::vformat as native_vertex_format;
mod layout;
mod raster;
pub(crate) mod specialization;

pub use binding::{VulkanBindGroup, VulkanBindGroupLayout};
pub use cache::VulkanPipelineCache;
pub use compute::VulkanComputePipeline;
pub(crate) use layout::PushConstants;
pub use layout::VulkanPipelineLayout;
pub use raster::VulkanRasterPipeline;

pub(crate) use binding::{create_group, create_group_layout};
pub(crate) use compute::create_compute;
pub(crate) use layout::create_layout;
pub(crate) use raster::create_raster;
