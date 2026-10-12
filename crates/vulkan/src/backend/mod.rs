//! Vulkan instance and device selection.
//!
//! This module establishes real Vulkan ownership.  Resource, pipeline, WSI
//! and encoder lowerings are split into sibling modules as they are added;
//! no renderer scheduler or CPU command replay layer lives here.

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

mod capture;
mod compute;
mod copy;
mod encoder;
mod pipeline;
mod query;
mod raster;
mod readback;
mod resource;
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
    max_draw_indirect_count: u32,
    /// Serializes Vulkan queue submission and fence allocation only. It is
    /// not a graph scheduler and must never span host completion waits.
    queue_gate: Mutex<()>,
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

impl VulkanInstance {
    /// Loads the Vulkan loader and creates a private API 1.3 instance.
    pub fn new() -> Result<Self> {
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
        let instance = unsafe { entry.create_instance(&info, None) }.map_err(vk_error)?;
        Ok(Self(Arc::new(InstanceState { entry, instance })))
    }

    /// Creates a typed Core instance with a fresh identity.
    pub fn instance(&self) -> Result<api::Instance<Vulkan>> {
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
        unsafe { surface::create_surface(device, window, display) }
    }
}

impl Drop for InstanceState {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}

impl Drop for DeviceState {
    fn drop(&mut self) {
        // Device teardown is only reached after all native resource Arcs have
        // released their DeviceState owner.
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
        _: &mut VulkanEncoder,
        _: api::TextureRegion<'_, Self>,
        _: api::TextureRegion<'_, Self>,
    ) -> Result<()> {
        Err(Error::Unsupported("Vulkan texture copy"))
    }
    fn buffer_to_texture(
        _: &mut VulkanEncoder,
        _: &VulkanBuffer,
        _: api::TexelCopyBufferLayout,
        _: api::TextureRegion<'_, Self>,
    ) -> Result<()> {
        Err(Error::Unsupported("Vulkan buffer-to-texture"))
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

impl InstanceBackend for Vulkan {
    type NativeInstance = VulkanInstance;
    type OpenFuture<'a> = Ready<Result<(VulkanDevice, DeviceCapabilities)>>;

    fn enumerate(instance: &VulkanInstance, scope: InstanceId) -> Result<Vec<AdapterInfo>> {
        let physical =
            unsafe { instance.0.instance.enumerate_physical_devices() }.map_err(vk_error)?;
        let adapters = physical
            .into_iter()
            .enumerate()
            .filter_map(|(ordinal, physical)| {
                let families = unsafe {
                    instance
                        .0
                        .instance
                        .get_physical_device_queue_family_properties(physical)
                };
                families
                    .iter()
                    .any(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                    .then(|| {
                        let properties =
                            unsafe { instance.0.instance.get_physical_device_properties(physical) };
                        let name =
                            unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()) }
                                .to_string_lossy()
                                .into_owned();
                        AdapterInfo {
                            id: AdapterId {
                                instance: scope,
                                ordinal: ordinal as u64,
                            },
                            name,
                        }
                    })
            })
            .collect();
        Ok(adapters)
    }

    fn open<'a>(
        instance: &'a VulkanInstance,
        device: DeviceId,
        request: &'a DeviceRequest,
    ) -> Self::OpenFuture<'a> {
        ready((|| {
            if !request.required_features.is_empty() {
                return Err(Error::Unsupported("requested Vulkan device features"));
            }
            let physical =
                unsafe { instance.0.instance.enumerate_physical_devices() }.map_err(vk_error)?;
            let physical = physical
                .get(request.adapter.ordinal as usize)
                .copied()
                .ok_or(Error::Unsupported("the requested Vulkan adapter"))?;
            let family = unsafe {
                instance
                    .0
                    .instance
                    .get_physical_device_queue_family_properties(physical)
            }
            .iter()
            .position(|f| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .ok_or(Error::Unsupported("a Vulkan graphics queue"))? as u32;
            let (dynamic_rendering, shader_clip_distance, multi_draw_indirect) = {
                let mut available13 = vk::PhysicalDeviceVulkan13Features::default();
                let mut available =
                    vk::PhysicalDeviceFeatures2::default().push_next(&mut available13);
                unsafe {
                    instance
                        .0
                        .instance
                        .get_physical_device_features2(physical, &mut available)
                };
                let shader_clip_distance = available.features.shader_clip_distance == vk::TRUE;
                let multi_draw_indirect = available.features.multi_draw_indirect == vk::TRUE;
                let dynamic_rendering = available13.dynamic_rendering == vk::TRUE;
                (dynamic_rendering, shader_clip_distance, multi_draw_indirect)
            };
            if !dynamic_rendering {
                return Err(Error::Unsupported("Vulkan dynamic rendering"));
            }
            let enabled_features = vk::PhysicalDeviceFeatures::default()
                .shader_clip_distance(shader_clip_distance)
                .multi_draw_indirect(multi_draw_indirect);
            let priorities = [1.0_f32];
            let queue = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priorities);
            let extensions = [ash::khr::swapchain::NAME.as_ptr()];
            let mut vulkan13 =
                vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
            let create = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue))
                .enabled_extension_names(&extensions)
                .enabled_features(&enabled_features)
                .push_next(&mut vulkan13);
            let native = unsafe { instance.0.instance.create_device(physical, &create, None) }
                .map_err(vk_error)?;
            let graphics_queue = unsafe { native.get_device_queue(family, 0) };
            let properties =
                unsafe { instance.0.instance.get_physical_device_properties(physical) };
            let limits = properties.limits;
            let rgba8_features = unsafe {
                instance
                    .0
                    .instance
                    .get_physical_device_format_properties(physical, vk::Format::R8G8B8A8_UNORM)
            }
            .optimal_tiling_features;
            Ok((
                VulkanDevice(Arc::new(DeviceState {
                    device: native,
                    instance: instance.0.clone(),
                    physical,
                    graphics_queue,
                    graphics_family: family,
                    shader_clip_distance,
                    multi_draw_indirect,
                    max_draw_indirect_count: limits.max_draw_indirect_count,
                    queue_gate: Mutex::new(()),
                })),
                DeviceCapabilities {
                    compute: true,
                    indirect_draw: true,
                    multi_draw: multi_draw_indirect,
                    storage_textures: rgba8_features
                        .contains(vk::FormatFeatureFlags::STORAGE_IMAGE),
                    push_constants: limits.max_push_constants_size != 0,
                    lanes: Arc::from([LaneInfo {
                        id: LaneId { device, ordinal: 0 },
                        class: LaneClass::Graphics,
                        routes: Vec::new(),
                    }]),
                    limits: DeviceLimits {
                        max_bind_groups: 4,
                        max_bindings_per_group: 1024,
                        max_color_attachments: limits.max_color_attachments,
                        max_texture_dimension_2d: limits.max_image_dimension2_d,
                        max_texture_array_layers: limits.max_image_array_layers,
                        max_buffer_bytes: limits.max_storage_buffer_range as u64,
                        max_vertex_buffers: limits.max_vertex_input_bindings,
                        max_vertex_attributes: limits.max_vertex_input_attributes,
                        max_push_constant_bytes: limits.max_push_constants_size,
                        uniform_offset_alignment: limits.min_uniform_buffer_offset_alignment,
                        storage_offset_alignment: limits.min_storage_buffer_offset_alignment,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            ))
        })())
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
