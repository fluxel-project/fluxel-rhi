//! Device facts for implemented Metal operations and their native limits.
//!
//! Logical group indices do not reserve native slots. Pipeline creation checks
//! the combined per-stage resource budget after assigning actual Metal slots.
use super::{Metal, MetalDevice, binding_abi, bindings, memory, mesh, shader, sparse, texture};
use fluxel_rhi_core::api::{DeviceCapabilities, DeviceLimits, QueryKind};
use fluxel_rhi_core::backend::QueryBackend;
use objc2::runtime::NSObjectProtocol as _;
use objc2_metal::MTLDevice as _;

pub(super) fn describe(device: &MetalDevice, multi_queue: bool) -> DeviceCapabilities {
    let native = &device._state._device;
    let threads = native.maxThreadsPerThreadgroup();
    let multiview = native.respondsToSelector(objc2::sel!(supportsVertexAmplificationCount:))
        && native.supportsVertexAmplificationCount(2);
    DeviceCapabilities {
        // Metal exposes annotations on native render/compute encoders. It has
        // no equivalent command-buffer marker in this backend's direct path.
        debug_markers: true,
        debug_markers_encoder: false,
        debug_markers_raster_pass: true,
        debug_markers_compute_pass: true,
        compute: true,
        sparse: sparse::placement_supported(device),
        mesh: mesh::mesh_supported(device),
        acceleration_structure: device._state.ray_supported,
        // The converter SDK route is intentionally not part of this backend.
        ray: false,
        ray_query: shader::ray_query_supported(device),
        multi_queue,
        indirect_draw: true,
        indirect_first_instance: true,
        multi_draw: true,
        multi_draw_count: device._state.indirect_count_supported,
        push_constants: true,
        storage_buffers: true,
        storage_textures: true,
        binding_arrays: bindings::argument_arrays_supported(native),
        runtime_sized_binding_arrays: bindings::argument_arrays_supported(native),
        nonuniform_sampled_texture_and_storage_buffer_indexing: bindings::argument_arrays_supported(
            native,
        ),
        timestamp: Metal::query_supported(device, QueryKind::Timestamp),
        occlusion: Metal::query_supported(device, QueryKind::Occlusion),
        multiview,
        placed_memory: memory::supported(native),
        external_memory: true,
        native_interop: true,
        // Metal render encoders expose depth bias as mutable encoder state.
        // The command accepts a clamp factor on all supported Metal devices.
        dynamic_depth_bias: true,
        dynamic_depth_bias_clamp: true,
        limits: DeviceLimits {
            // Logical indices do not allocate native slots. These values are
            // index-space bounds, not a guarantee for every resource mixture;
            // pipeline creation checks combined per-stage Metal slot budgets.
            max_bind_groups: u32::MAX,
            max_bindings_per_group: u32::MAX,
            max_color_attachments: 8,
            max_texture_dimension_2d: texture::max_texture_dimension_2d(native),
            max_texture_array_layers: 2048,
            max_buffer_bytes: native.maxBufferLength() as u64,
            max_vertex_buffers: binding_abi::MAX_VERTEX_BUFFERS as u32,
            max_vertex_attributes: binding_abi::MAX_VERTEX_ATTRIBUTES,
            max_compute_workgroups: [u32::MAX; 3],
            max_compute_workgroup_size: [
                threads.width as u32,
                threads.height as u32,
                threads.depth as u32,
            ],
            max_push_constant_bytes: (native.maxBufferLength().min(u32::MAX as usize) as u32) & !3,
            // Native amplification batches remap sparse masks across 32 views.
            max_multiview_views: if multiview { 32 } else { 0 },
            uniform_offset_alignment: bindings::uniform_offset_alignment(native),
            storage_offset_alignment: 4,
        },
        ..DeviceCapabilities::default()
    }
}
