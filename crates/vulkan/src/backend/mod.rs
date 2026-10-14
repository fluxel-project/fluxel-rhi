//! Vulkan instance and device selection.
//!
//! This module establishes real Vulkan ownership.  Resource, pipeline, WSI
//! and encoder lowerings are split into sibling modules as they are added;
//! no renderer scheduler or CPU command replay layer lives here.
//! `InstanceState` outlives every `DeviceState`, and each DeviceState outlives
//! all private native children through retained `Arc`s. This root selects one
//! graphics queue and reports its capabilities; submission, resource state and
//! WSI policy are owned by their narrow sibling modules.

use std::{
    future::{Ready, ready},
    sync::{Arc, Mutex},
};

use ash::{Entry, Instance, vk};
use fluxel_rhi_core::api::{
    self, AdapterId, AdapterInfo, Backend, DeviceCapabilities, DeviceId, DeviceLimits,
    DeviceRequest, Error, InstanceBackend, InstanceId, LaneClass, LaneId, LaneInfo, Result,
};

/// Vulkan backend selected once by the caller.
pub struct Vulkan;

/// Process-local Vulkan instance used for adapter enumeration and opening.
#[derive(Clone)]
pub struct VulkanInstance(Arc<InstanceState>);

struct InstanceState {
    entry: Entry,
    instance: Instance,
}

mod instance;

mod capture;
mod compute;
mod copy;
mod encoder;
mod pipeline;
mod query;
mod raster;
mod readback;
mod resource;
mod secondary;
mod shader;
mod submit;
mod surface;
mod transition;

/// Logical Vulkan device and its one direct graphics queue.
#[derive(Clone)]
pub struct VulkanDevice(Arc<DeviceState>);

struct DeviceState {
    device: ash::Device,
    // The device may outlive the typed VulkanInstance returned by its creator.
    // Retaining this owner keeps the dispatch table and VkInstance alive until
    // after `vkDestroyDevice`.
    instance: Arc<InstanceState>,
    physical: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    graphics_family: u32,
    shader_clip_distance: bool,
    multi_draw_indirect: bool,
    /// Enabled only when the physical device supports sampler anisotropy.
    sampler_anisotropy: bool,
    max_sampler_anisotropy: f32,
    /// Enabled at device creation, so dynamic rendering may broadcast one
    /// graphics invocation to the layers selected by `RenderingInfo::view_mask`.
    multiview: bool,
    /// The descriptor-indexing feature subset required by the public binding
    /// array contract.  Layout creation is deliberately gated by Core
    /// capabilities; retaining this fact here keeps native lowering honest.
    descriptor_indexing: bool,
    max_sampled_images: u32,
    max_draw_indirect_count: u32,
    /// Serializes Vulkan queue submission and fence allocation only. It is
    /// not a graph scheduler and must never span host completion waits.
    queue_gate: Mutex<()>,
    /// Keeps one Core submission plan contiguous on the native queue.  It is
    /// separate from `queue_gate`: presentation and other one-shot queue
    /// calls still use the narrower gate, while a plan never interleaves its
    /// explicit resource-state sequence with another submitter.
    submit_gate: Mutex<()>,
}

macro_rules! native_arc {
    ($($name:ident),* $(,)?) => {$(
        #[derive(Clone)]
        pub struct $name(Arc<()>);
    )*};
}

native_arc!(
    VulkanMeshPipeline,
    VulkanRayPipeline,
    VulkanPipelineCache,
    VulkanAcceleration,
    VulkanSparse,
);

pub use pipeline::{
    VulkanBindGroup, VulkanBindGroupLayout, VulkanComputePipeline, VulkanPipelineLayout,
    VulkanRasterPipeline,
};
pub use query::VulkanQuerySet;
pub use resource::{VulkanBuffer, VulkanSampler, VulkanTexture, VulkanTextureView};
pub use shader::VulkanShader;
pub use surface::{VulkanSurface, VulkanSurfaceImage};

/// Native command encoder. Its concrete command-pool/list state is added by
/// `encoder`; keeping it opaque here prevents a catch-all compatibility file.
pub use encoder::VulkanEncoder;
pub use secondary::VulkanSecondaryWork;

impl VulkanInstance {
    /// Loads the Vulkan loader and creates a private API 1.3 instance.
    pub fn new() -> Result<Self> {
        // SAFETY: loading the process Vulkan loader creates a dispatch table
        // owned by InstanceState and used only while that state remains alive.
        let entry = unsafe { Entry::load() }
            .map_err(|error| Error::Backend(format!("Vulkan loader: {error}")))?;
        let app = vk::ApplicationInfo::default()
            .application_name(c"Fluxel RHI")
            .application_version(0)
            .engine_name(c"Fluxel")
            .engine_version(0)
            .api_version(vk::API_VERSION_1_3);
        #[cfg(windows)]
        let extensions = [
            ash::khr::surface::NAME.as_ptr(),
            ash::khr::win32_surface::NAME.as_ptr(),
        ];
        #[cfg(not(windows))]
        let extensions = [ash::khr::surface::NAME.as_ptr()];
        let info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&extensions);
        // SAFETY: all extension name pointers are static Vulkan names and the
        // create info borrows them only for this call.
        let instance = unsafe { entry.create_instance(&info, None) }.map_err(vk_error)?;
        Ok(Self(Arc::new(InstanceState { entry, instance })))
    }

    /// Creates a typed Core instance with a fresh identity.
    pub fn instance(&self) -> Result<api::Instance<Vulkan>> {
        // SAFETY: `self` is the native instance whose lifetime is retained by
        // the returned typed Core instance and the allocated identity is fresh.
        Ok(unsafe { api::Instance::from_native(InstanceId::allocate()?, self.clone()) })
    }

    /// Opens the first graphics-capable physical device.
    pub async fn open_default(&self) -> Result<api::Device<Vulkan>> {
        let instance = self.instance()?;
        let adapter = instance
            .enumerate()?
            .into_iter()
            .next()
            .ok_or(Error::Unsupported("a Vulkan graphics adapter"))?;
        instance
            .open(DeviceRequest {
                adapter: adapter.id,
                required_features: Vec::new(),
            })
            .await
    }
}

impl Vulkan {
    /// Creates a Vulkan WSI surface for a Win32 window owned by the caller.
    ///
    /// # Safety
    /// Both raw handles remain valid until the surface and every image lease
    /// originating from it have been destroyed.
    pub unsafe fn surface_from_raw_window_handle(
        device: &api::Device<Self>,
        window: raw_window_handle::RawWindowHandle,
        display: raw_window_handle::RawDisplayHandle,
    ) -> Result<api::Surface<Self>> {
        // SAFETY: forwarded unchanged; the caller's handle-lifetime contract
        // is exactly the one required by the private WSI owner.
        unsafe { surface::create_surface(device, window, display) }
    }
}

impl Drop for InstanceState {
    fn drop(&mut self) {
        // SAFETY: DeviceState retains InstanceState, so no logical device can
        // outlive this final instance owner.
        unsafe { self.instance.destroy_instance(None) };
    }
}

impl Drop for DeviceState {
    fn drop(&mut self) {
        // Device teardown is only reached after all native resource Arcs have
        // released their DeviceState owner.
        // SAFETY: resource, encoder, surface and completion Arcs all retain
        // DeviceState, so no child Vulkan object can outlive this destroy call.
        unsafe { self.device.destroy_device(None) };
    }
}

impl Backend for Vulkan {
    type NativeDevice = VulkanDevice;
    type NativeBuffer = VulkanBuffer;
    type NativeTexture = VulkanTexture;
    type NativeTextureView = VulkanTextureView;
    type NativeSampler = VulkanSampler;
    type NativeBindGroupLayout = VulkanBindGroupLayout;
    type NativeBindGroup = VulkanBindGroup;
    type NativePipelineLayout = VulkanPipelineLayout;
    type NativeShader = VulkanShader;
    type NativeRasterPipeline = VulkanRasterPipeline;
    type NativeComputePipeline = VulkanComputePipeline;
    type NativeMeshPipeline = VulkanMeshPipeline;
    type NativeRayPipeline = VulkanRayPipeline;
    type NativePipelineCache = VulkanPipelineCache;
    type NativeAcceleration = VulkanAcceleration;
    type NativeSparseAllocation = VulkanSparse;
    type NativeQuerySet = VulkanQuerySet;
    type NativeEncoder = VulkanEncoder;
    type NativeCompletion = submit::Completion;
    type NativeSurface = VulkanSurface;
    type NativeSurfaceImage = VulkanSurfaceImage;
    type NativeUncertainty = ();
    type Compute = compute::VulkanCompute;
    type Mesh = api::Unavailable;
    type Ray = api::Unavailable;
    type Extras = raster::RasterExtras;
}

impl api::ResourceBackend for Vulkan {
    fn buffer_support(
        device: &VulkanDevice,
        desc: &api::BufferDesc,
    ) -> api::Support<api::BufferSupport> {
        resource::buffer_support(device, desc)
    }
    fn texture_support(
        device: &VulkanDevice,
        desc: &api::TextureDesc,
    ) -> api::Support<api::TextureSupport> {
        resource::texture_support(device, desc)
    }
    fn create_buffer(device: &VulkanDevice, desc: &api::BufferDesc) -> Result<VulkanBuffer> {
        resource::create_buffer(device, desc)
    }
    fn create_texture(device: &VulkanDevice, desc: &api::TextureDesc) -> Result<VulkanTexture> {
        resource::create_texture(device, desc)
    }
    fn create_view(
        device: &VulkanDevice,
        texture: &VulkanTexture,
        desc: &api::TextureViewDesc,
    ) -> Result<VulkanTextureView> {
        resource::create_view(device, texture, desc)
    }
    fn create_sampler(device: &VulkanDevice, desc: &api::SamplerDesc) -> Result<VulkanSampler> {
        resource::create_sampler(device, desc)
    }
    fn create_bind_group_layout(
        device: &VulkanDevice,
        desc: &api::BindGroupLayoutDesc,
    ) -> Result<VulkanBindGroupLayout> {
        pipeline::create_group_layout(device, desc)
    }
    fn create_bind_group(
        device: &VulkanDevice,
        desc: &api::BindGroupDesc<'_, Self>,
    ) -> Result<VulkanBindGroup> {
        pipeline::create_group(device, desc)
    }
    fn create_pipeline_layout(
        device: &VulkanDevice,
        desc: &api::PipelineLayoutDesc<'_, Self>,
    ) -> Result<VulkanPipelineLayout> {
        pipeline::create_layout(device, desc)
    }
}

impl api::PipelineBackend for Vulkan {
    type RasterPipelineFuture<'a> = Ready<Result<VulkanRasterPipeline>>;
    fn create_raster_pipeline<'a>(
        device: &'a VulkanDevice,
        desc: &'a api::RasterPipelineDesc<Self>,
    ) -> Self::RasterPipelineFuture<'a> {
        ready(pipeline::create_raster(device, desc))
    }
}

impl api::QueryBackend for Vulkan {
    fn query_supported(_: &VulkanDevice, kind: api::QueryKind) -> bool {
        matches!(kind, api::QueryKind::Occlusion)
    }

    fn create_query_set(device: &VulkanDevice, desc: &api::QueryDesc) -> Result<VulkanQuerySet> {
        query::create(device, desc)
    }

    fn write_timestamp(_: &mut VulkanEncoder, _: &VulkanQuerySet, _: u32) -> Result<()> {
        Err(Error::Unsupported("Vulkan timestamps"))
    }

    fn begin_occlusion(encoder: &mut VulkanEncoder, set: &VulkanQuerySet, slot: u32) -> Result<()> {
        query::begin_occlusion(encoder, set, slot)
    }

    fn reset_queries(
        encoder: &mut VulkanEncoder,
        set: &VulkanQuerySet,
        slots: std::ops::Range<u32>,
    ) -> Result<()> {
        query::reset(encoder, set, slots)
    }

    fn end_occlusion(encoder: &mut VulkanEncoder) -> Result<()> {
        query::end_occlusion(encoder)
    }

    fn begin_statistics(_: &mut VulkanEncoder, _: &VulkanQuerySet, _: u32) -> Result<()> {
        Err(Error::Unsupported("Vulkan pipeline statistics"))
    }

    fn end_statistics(_: &mut VulkanEncoder) -> Result<()> {
        Err(Error::Unsupported("Vulkan pipeline statistics"))
    }

    fn resolve_queries(
        encoder: &mut VulkanEncoder,
        set: &VulkanQuerySet,
        slots: std::ops::Range<u32>,
        destination: &VulkanBuffer,
        offset: u64,
    ) -> Result<()> {
        query::resolve(encoder, set, slots, destination, offset)
    }
}

impl api::CommandBackend for Vulkan {
    fn begin_encoder(device: &VulkanDevice, _: LaneId) -> Result<VulkanEncoder> {
        VulkanEncoder::new(device)
    }
    fn transition(e: &mut VulkanEncoder, ops: &[api::ResourceTransition<'_, Self>]) -> Result<()> {
        transition::record(e, ops)
    }
    fn copy_buffer(
        e: &mut VulkanEncoder,
        src: &VulkanBuffer,
        so: u64,
        dst: &VulkanBuffer,
        doff: u64,
        bytes: u64,
    ) -> Result<()> {
        copy::copy_buffer(e, src, so, dst, doff, bytes)
    }
    fn copy_texture(
        e: &mut VulkanEncoder,
        src: api::TextureRegion<'_, Self>,
        dst: api::TextureRegion<'_, Self>,
    ) -> Result<()> {
        copy::copy_texture(e, src, dst)
    }
    fn buffer_to_texture(
        e: &mut VulkanEncoder,
        src: &VulkanBuffer,
        layout: api::TexelCopyBufferLayout,
        dst: api::TextureRegion<'_, Self>,
    ) -> Result<()> {
        copy::buffer_to_texture(e, src, layout, dst)
    }
    fn texture_to_buffer(
        encoder: &mut VulkanEncoder,
        source: api::TextureRegion<'_, Self>,
        destination: &VulkanBuffer,
        layout: api::TexelCopyBufferLayout,
    ) -> Result<()> {
        copy::texture_to_buffer(encoder, source, destination, layout)
    }
    fn write_buffer(
        e: &mut VulkanEncoder,
        dst: &VulkanBuffer,
        offset: u64,
        bytes: &[u8],
    ) -> Result<()> {
        copy::write_buffer(e, dst, offset, bytes)
    }
    fn write_texture(
        e: &mut VulkanEncoder,
        region: api::TextureRegion<'_, Self>,
        layout: api::TexelCopyBufferLayout,
        bytes: &[u8],
    ) -> Result<()> {
        copy::write_texture(e, region, layout, bytes)
    }
    fn blit(
        e: &mut VulkanEncoder,
        src: api::TextureRegion<'_, Self>,
        dst: api::TextureRegion<'_, Self>,
        filter: api::FilterMode,
    ) -> Result<()> {
        copy::blit(e, src, dst, filter)
    }
}

impl api::ShaderBackend for Vulkan {
    type DirectFuture<'a> = Ready<Result<VulkanShader>>;
    type NagaFuture<'a> = Ready<Result<VulkanShader>>;
    fn shader_route(_: &VulkanDevice, desc: &api::ShaderDesc) -> api::ShaderRoute {
        match desc.format {
            api::ShaderFormat::SpirV => api::ShaderRoute::Direct(api::ShaderTarget {
                format: api::ShaderFormat::SpirV,
                environment: "vulkan-1.3".into(),
                abi_key: "spirv-vulkan-set-binding".into(),
            }),
            _ => api::ShaderRoute::Unsupported,
        }
    }
    fn compile_direct<'a>(
        device: &'a VulkanDevice,
        desc: &'a api::ShaderDesc,
        _: &'a api::ShaderTarget,
    ) -> Self::DirectFuture<'a> {
        ready(shader::create(device, desc))
    }
    fn compile_naga<'a>(
        _: &'a VulkanDevice,
        _: &'a api::ShaderDesc,
        _: &'a api::ShaderTarget,
    ) -> Self::NagaFuture<'a> {
        ready(Err(Error::Unsupported("Vulkan Naga shader route")))
    }
}

fn vk_error(error: vk::Result) -> Error {
    match error {
        vk::Result::ERROR_DEVICE_LOST => Error::DeviceLost,
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            Error::OutOfMemory
        }
        vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR => Error::SurfaceOutdated,
        vk::Result::ERROR_SURFACE_LOST_KHR => Error::SurfaceLost,
        _ => Error::Backend(format!("Vulkan: {error:?}")),
    }
}
