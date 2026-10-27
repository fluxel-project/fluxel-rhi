//! Vulkan instance and device selection.
//!
//! This module establishes real Vulkan ownership.  Resource, pipeline, WSI
//! and encoder lowerings are split into sibling modules as they are added;
//! no renderer scheduler or CPU command replay layer lives here.
//! `InstanceState` outlives every `DeviceState`, and each DeviceState outlives
//! all private native children through retained `Arc`s. This root selects one
//! graphics queue and an optional compute queue and reports their capabilities; submission, resource state and
//! WSI policy are owned by their narrow sibling modules.

use std::{
    future::{Ready, ready},
    sync::{Arc, Mutex},
};

use ash::{Entry, Instance, vk};
use fluxel_rhi_core::api::{
    self, AdapterId, AdapterInfo, DeviceCapabilities, DeviceId, DeviceLimits, DeviceRequest, Error,
    InstanceId, LaneClass, LaneId, LaneInfo, Result,
};
use fluxel_rhi_core::backend::{Backend, InstanceBackend};

/// Vulkan backend selected once by the caller.
#[derive(Clone)]
pub struct Vulkan;

#[cfg(all(test, feature = "hardware-tests"))]
#[path = "../tests/backend/queue_ownership_gpu.rs"]
mod queue_ownership_gpu;

/// Process-local Vulkan instance used for adapter enumeration and opening.
#[derive(Clone)]
pub struct VulkanInstance(Arc<InstanceState>);

struct InstanceState {
    entry: Entry,
    instance: Instance,
    /// `VK_EXT_debug_utils` was enabled when this instance was created.
    debug_utils_enabled: bool,
    #[cfg(all(unix, not(target_os = "android")))]
    surface_extensions: [bool; 3],
}

mod external;
mod instance;
mod interop;
pub use external::{VulkanExternalHandle, VulkanExternalMemoryHandle, VulkanExternalSync};

mod capture;
mod clear;
mod cold_worker;
mod command_pool;
mod compute;
mod copy;
mod encoder;
mod mapping;
mod memory;
mod mesh;
mod pipeline;
mod query;
mod queue_support;
mod raster;
mod ray;
mod readback;
mod resource;
mod secondary;
mod shader;
mod sparse;
mod submit;
mod surface;
mod transition;
mod upload;

/// Logical Vulkan device and its exposed native queues.
#[derive(Clone)]
pub struct VulkanDevice(Arc<DeviceState>);

/// Logical-device shader precision and storage features actually enabled.
///
/// The direct SPIR-V route inspects only capabilities whose Vulkan feature is
/// explicit. Keeping copied booleans instead of feature-chain nodes prevents
/// capability admission from observing physical support that was not enabled.
#[derive(Clone, Copy)]
struct ShaderPrecisionFeatures {
    float16: bool,
    int8: bool,
    storage_buffer16: bool,
    uniform_and_storage_buffer16: bool,
    storage_push_constant16: bool,
    storage_input_output16: bool,
    storage_buffer8: bool,
    uniform_and_storage_buffer8: bool,
    storage_push_constant8: bool,
}

struct DeviceState {
    upload_pool: upload::UploadPool,
    upload_copy_offset_alignment: u64,
    readback_pool: readback::ReadbackPool,
    default_resource_pool: memory::DefaultResourcePool,
    command_pools: command_pool::CommandPoolCache,
    descriptor_pools: pipeline::descriptor_pool::DescriptorPoolCache,
    /// Base features actually enabled on the logical device.
    features: vk::PhysicalDeviceFeatures,
    shader_draw_index: bool,
    texture_compression_astc_hdr: bool,
    descriptor_partially_bound: bool,
    nonuniform_uniform_buffer: bool,
    nonuniform_sampled_image: bool,
    nonuniform_storage_buffer: bool,
    nonuniform_storage_image: bool,
    runtime_descriptor_array: bool,
    variable_descriptor_count: bool,
    separate_depth_stencil_layouts: bool,
    subgroup_size_control: bool,
    min_subgroup_size: u32,
    max_subgroup_size: u32,
    required_subgroup_size_stages: vk::ShaderStageFlags,
    subgroup_stages: vk::ShaderStageFlags,
    subgroup_operations: vk::SubgroupFeatureFlags,
    subgroup_quad_operations_in_all_stages: bool,
    shader_precision: ShaderPrecisionFeatures,
    conservative_rasterization: bool,
    maintenance5: bool,
    external_memory_handles: bool,
    external_semaphore_handles: bool,
    compute_family: u32,
    compute_timestamp_bits: u32,
    compute_sparse_binding: bool,
    graphics_sparse_binding: bool,
    graphics_compute: bool,
    max_compute_workgroups: [u32; 3],
    max_compute_workgroup_size: [u32; 3],
    max_compute_workgroup_invocations: u32,
    /// Raw device-owned queue timelines. Wrappers retained by submitted jobs
    /// point back to DeviceState but never own these semaphores, so reuse does
    /// not form a State -> wrapper -> State cycle.
    queue_timelines: Option<[submit::queue_timeline::RawQueueTimeline; 2]>,
    device: ash::Device,
    // The device may outlive the typed VulkanInstance returned by its creator.
    // Retaining this owner keeps the dispatch table and VkInstance alive until
    // after `vkDestroyDevice`.
    instance: Arc<InstanceState>,
    physical: vk::PhysicalDevice,
    graphics_queue: vk::Queue,
    graphics_family: u32,
    compute_queue: Option<vk::Queue>,
    mesh: Option<mesh::MeshDevice>,
    ray_support: ray::probe::RaySupport,
    ray_loaders: Option<ray::probe::RayLoaders>,
    sparse_image: bool,
    timeline_semaphore: bool,
    shader_clip_distance: bool,
    multi_draw_indirect: bool,
    draw_indirect_count: bool,
    pipeline_statistics: bool,
    timestamp_period: f64,
    timestamp_bits: u32,
    /// Enabled only when the physical device supports sampler anisotropy.
    sampler_anisotropy: bool,
    max_sampler_anisotropy: f32,
    /// Enabled at device creation, so dynamic rendering may broadcast one
    /// graphics invocation to the layers selected by `RenderingInfo::view_mask`.
    multiview: bool,
    max_multiview_views: u32,
    multiview_mesh_shader: bool,
    max_draw_indirect_count: u32,
    /// Loaded only after all command-label entry points were verified.
    debug_utils: Option<ash::ext::debug_utils::Device>,
    /// Serializes Vulkan queue submission and fence allocation only. It is
    /// not a graph scheduler and must never span host completion waits.
    queue_gate: Mutex<()>,
    /// Keeps one Core submission plan contiguous on the native queue.  It is
    /// separate from `queue_gate`: presentation and other one-shot queue
    /// calls still use the narrower gate, while a plan never interleaves its
    /// explicit resource-state sequence with another submitter.
    submit_gate: Mutex<()>,
}

pub use pipeline::VulkanPipelineCache;

pub use sparse::VulkanSparse;

pub use mesh::VulkanMeshPipeline;
pub use pipeline::{
    VulkanBindGroup, VulkanBindGroupLayout, VulkanComputePipeline, VulkanPipelineLayout,
    VulkanRasterPipeline,
};
pub use query::VulkanQuerySet;
pub use ray::{VulkanAcceleration, VulkanRayPipeline};
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
        let mut extensions = vec![
            ash::khr::surface::NAME.as_ptr(),
            ash::khr::win32_surface::NAME.as_ptr(),
        ];
        #[cfg(target_os = "android")]
        let mut extensions = vec![
            ash::khr::surface::NAME.as_ptr(),
            ash::khr::android_surface::NAME.as_ptr(),
        ];
        #[cfg(not(any(windows, target_os = "android")))]
        let mut extensions = vec![ash::khr::surface::NAME.as_ptr()];
        // Enumeration can change between the loader's count and data queries.
        // ash retries an incomplete data query, but not an incomplete count
        // query. Concurrent instance creation reproduced the latter on Windows.
        // Retry this read-only operation with a bound; never retry device or
        // resource creation here.
        let mut attempts = 0;
        let available = loop {
            // SAFETY: loader output owns each extension name for this query.
            match unsafe { entry.enumerate_instance_extension_properties(None) } {
                Ok(available) => break available,
                Err(vk::Result::INCOMPLETE) if attempts < 7 => attempts += 1,
                Err(error) => {
                    return Err(Error::Backend(format!(
                        "Vulkan instance extension enumeration: {error:?}"
                    )));
                }
            }
        };
        if available.iter().any(|extension| {
            // SAFETY: Vulkan guarantees NUL termination of extension_name.
            (unsafe { std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) })
                == ash::ext::swapchain_colorspace::NAME
        }) {
            extensions.push(ash::ext::swapchain_colorspace::NAME.as_ptr());
        }
        let debug_utils_enabled = available.iter().any(|extension| {
            // SAFETY: Vulkan guarantees NUL termination of extension names.
            (unsafe { std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) })
                == ash::ext::debug_utils::NAME
        });
        if debug_utils_enabled {
            extensions.push(ash::ext::debug_utils::NAME.as_ptr());
        }
        #[cfg(all(unix, not(target_os = "android")))]
        let surface_extensions = [
            ash::khr::xlib_surface::NAME,
            ash::khr::xcb_surface::NAME,
            ash::khr::wayland_surface::NAME,
        ]
        .map(|name| {
            let supported = available.iter().any(|extension| {
                // SAFETY: Vulkan extension names are NUL terminated.
                unsafe { std::ffi::CStr::from_ptr(extension.extension_name.as_ptr()) == name }
            });
            if supported {
                extensions.push(name.as_ptr());
            }
            supported
        });
        let info = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&extensions);
        // SAFETY: all extension name pointers are static Vulkan names and the
        // create info borrows them only for this call.
        let instance = unsafe { entry.create_instance(&info, None) }.map_err(vk_error)?;
        Ok(Self(Arc::new(InstanceState {
            entry,
            instance,
            debug_utils_enabled,
            #[cfg(all(unix, not(target_os = "android")))]
            surface_extensions,
        })))
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
    /// Creates a Vulkan WSI surface for a native window owned by the caller.
    ///
    /// # Safety
    /// Both raw handles remain valid until the surface and every image lease
    /// originating from it have been destroyed. Android callers must retain
    /// the `ANativeWindow` for the same interval.
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
        self.upload_pool.clear();
        self.readback_pool.clear();
        self.default_resource_pool.clear();
        self.command_pools.clear();
        self.descriptor_pools.clear();
        if let Some(timelines) = self.queue_timelines.take() {
            for timeline in timelines {
                // SAFETY: no wrapper outlives this final DeviceState owner;
                // each raw semaphore belongs to this logical device.
                unsafe { self.device.destroy_semaphore(timeline.semaphore, None) };
            }
        }
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
    type Mesh = mesh::Mesh;
    type Ray = ray::VulkanRay;
    type Extras = raster::RasterExtras;
    type MemoryObservation = memory::MemoryObservation;
}

impl fluxel_rhi_core::backend::ResourceBackend for Vulkan {
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

impl fluxel_rhi_core::backend::PipelineBackend for Vulkan {
    type RasterPipelineFuture<'a> = cold_worker::ColdFuture<VulkanRasterPipeline>;
    fn create_raster_pipeline<'a>(
        device: &'a VulkanDevice,
        desc: &'a api::RasterPipelineDesc<Self>,
        dynamic_depth_bias: Option<api::DynamicDepthBias>,
    ) -> Self::RasterPipelineFuture<'a> {
        let device = device.clone();
        let desc = desc.clone();
        cold_worker::submit(move || pipeline::create_raster(&device, &desc, dynamic_depth_bias))
    }
}

impl fluxel_rhi_core::backend::QueryBackend for Vulkan {
    fn query_supported(device: &VulkanDevice, kind: api::QueryKind) -> bool {
        match kind {
            api::QueryKind::Occlusion => true,
            api::QueryKind::Timestamp => {
                device.0.timestamp_bits != 0
                    || (device.0.compute_queue.is_some() && device.0.compute_timestamp_bits != 0)
            }
            api::QueryKind::PipelineStatistics => device.0.pipeline_statistics,
        }
    }
    fn timestamp_frequency(device: &VulkanDevice, lane: LaneId) -> Option<u64> {
        if Self::timestamp_clock(device, lane).is_none() {
            return None;
        }
        query::integer_frequency(device.0.timestamp_period as f32)
    }
    fn timestamp_clock(device: &VulkanDevice, lane: LaneId) -> Option<api::TimestampClock> {
        let valid_bits = match lane.ordinal {
            0 => device.0.timestamp_bits,
            1 if device.0.compute_queue.is_some() => device.0.compute_timestamp_bits,
            _ => return None,
        };
        if valid_bits == 0 {
            return None;
        }
        Some(api::TimestampClock {
            period_ns: device.0.timestamp_period,
            valid_bits,
        })
    }

    fn create_query_set(device: &VulkanDevice, desc: &api::QueryDesc) -> Result<VulkanQuerySet> {
        query::create(device, desc)
    }

    fn write_timestamp(e: &mut VulkanEncoder, set: &VulkanQuerySet, slot: u32) -> Result<()> {
        query::timestamp(e, set, slot)
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

    fn begin_statistics(e: &mut VulkanEncoder, set: &VulkanQuerySet, slot: u32) -> Result<()> {
        query::begin_statistics(e, set, slot)
    }
    fn end_statistics(e: &mut VulkanEncoder) -> Result<()> {
        query::end_statistics(e)
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

impl fluxel_rhi_core::backend::CommandBackend for Vulkan {
    fn debug_marker(
        encoder: &mut VulkanEncoder,
        _: fluxel_rhi_core::backend::DebugMarkerScope,
        op: fluxel_rhi_core::backend::DebugMarkerOp<'_>,
    ) -> Result<()> {
        use std::ffi::CString;

        let debug_utils = encoder
            .device
            .debug_utils
            .as_ref()
            .ok_or(Error::Unsupported("VK_EXT_debug_utils labels"))?;
        match op {
            fluxel_rhi_core::backend::DebugMarkerOp::Insert(text) => {
                let text = CString::new(text).map_err(|_| Error::InvalidRange)?;
                let label = vk::DebugUtilsLabelEXT::default().label_name(&text);
                // SAFETY: `debug_utils` was constructed only after all label
                // commands were resolved; command recording is exclusive and
                // `text`/`label` remain live for this synchronous call.
                unsafe { debug_utils.cmd_insert_debug_utils_label(encoder.command, &label) };
            }
            fluxel_rhi_core::backend::DebugMarkerOp::Push(text) => {
                let text = CString::new(text).map_err(|_| Error::InvalidRange)?;
                let label = vk::DebugUtilsLabelEXT::default().label_name(&text);
                // SAFETY: as above for insert.
                unsafe { debug_utils.cmd_begin_debug_utils_label(encoder.command, &label) };
                encoder.debug_group_depth = encoder
                    .debug_group_depth
                    .checked_add(1)
                    .ok_or(Error::InvalidState)?;
            }
            fluxel_rhi_core::backend::DebugMarkerOp::Pop => {
                if encoder.debug_group_depth == 0 {
                    return Err(Error::InvalidState);
                }
                // SAFETY: as above for insert; native depth was checked.
                unsafe { debug_utils.cmd_end_debug_utils_label(encoder.command) };
                encoder.debug_group_depth -= 1;
            }
        }
        Ok(())
    }

    fn clear_buffer(
        encoder: &mut VulkanEncoder,
        buffer: &VulkanBuffer,
        range: std::ops::Range<u64>,
        value: u32,
    ) -> Result<()> {
        clear::buffer(encoder, buffer, range, value)
    }
    fn clear_texture(
        e: &mut VulkanEncoder,
        texture: &VulkanTexture,
        mips: std::ops::Range<u32>,
        layers: std::ops::Range<u32>,
        value: api::TextureClearValue,
    ) -> Result<()> {
        clear::texture(e, texture, mips, layers, value)
    }
    fn begin_encoder(device: &VulkanDevice, lane: LaneId) -> Result<VulkanEncoder> {
        VulkanEncoder::new(device, lane)
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
    fn resolve(
        e: &mut VulkanEncoder,
        src: api::TextureRegion<'_, Self>,
        dst: api::TextureRegion<'_, Self>,
    ) -> Result<()> {
        copy::resolve(e, src, dst)
    }
}

impl fluxel_rhi_core::backend::ShaderBackend for Vulkan {
    type DirectFuture<'a> = cold_worker::ColdFuture<VulkanShader>;
    type NagaFuture<'a> = cold_worker::ColdFuture<VulkanShader>;
    fn shader_route(device: &VulkanDevice, desc: &api::ShaderDesc) -> api::ShaderRoute {
        if matches!(desc.format, api::ShaderFormat::Wgsl)
            && desc.required_features.iter().any(|feature| {
                feature == api::SHADER_FEATURE_PRIMITIVE_INDEX
                    || feature == api::SHADER_FEATURE_DRAW_INDEX
            })
        {
            // Naga 27 parses but does not implement WGSL's primitive_index
            // extension. Keep the feature on the direct SPIR-V route only.
            return api::ShaderRoute::Unsupported;
        }
        if desc
            .required_features
            .iter()
            .any(|feature| feature == api::SHADER_FEATURE_DRAW_INDEX)
            && !device.0.shader_draw_index
        {
            return api::ShaderRoute::Unsupported;
        }
        if desc
            .required_features
            .iter()
            .any(|feature| feature == api::SHADER_FEATURE_PRIMITIVE_INDEX)
            && device.0.features.geometry_shader != vk::TRUE
        {
            return api::ShaderRoute::Unsupported;
        }
        match desc.format {
            api::ShaderFormat::SpirV => api::ShaderRoute::Direct(api::ShaderTarget {
                format: api::ShaderFormat::SpirV,
                environment: "vulkan-1.3".into(),
                abi_key: "spirv-vulkan-set-binding".into(),
            }),
            api::ShaderFormat::Wgsl => api::ShaderRoute::Naga(api::ShaderTarget {
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
        let device = device.clone();
        let desc = desc.clone();
        cold_worker::submit(move || shader::create(&device, &desc))
    }
    fn compile_naga<'a>(
        device: &'a VulkanDevice,
        desc: &'a api::ShaderDesc,
        _: &'a api::ShaderTarget,
    ) -> Self::NagaFuture<'a> {
        let device = device.clone();
        let desc = desc.clone();
        cold_worker::submit(move || shader::create_wgsl(&device, &desc))
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

impl fluxel_rhi_core::backend::ExecutionBackend for Vulkan {
    type Mode = fluxel_rhi_core::api::DeferredExecution;
}
