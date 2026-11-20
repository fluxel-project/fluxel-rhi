//! Immutable capability and limit facts negotiated by a backend at device open.

use super::{CooperativeMatrixConfiguration, IndirectCountProfile, LaneInfo};
use std::sync::Arc;

/// Immutable facts negotiated at device open. Unsafe Device::from_native
/// requires these flags to match the implemented optional backend domains.
#[derive(Clone, Default)]
pub struct DeviceCapabilities {
    /// Pipeline layout descriptions can be derived from supported shader sources.
    pub shader_layout_inference: bool,
    /// Shader routes can use native half-precision floating-point values.
    pub shader_f16: bool,
    /// Native shaders can use exact signed and unsigned 16-bit integer types.
    pub shader_i16: bool,
    /// Native shaders can use double-precision floating-point values.
    pub shader_f64: bool,
    /// Native shaders can use 64-bit integers for ordinary computation.
    pub shader_int64: bool,
    /// Memory topology of the native node used by ordinary device work.
    /// This does not admit any particular buffer mapping or heap type.
    pub memory_topology: Option<DeviceMemoryTopology>,
    /// Supports explicit CPU-write allocations on a GPU upload heap.
    pub gpu_upload_buffers: bool,
    /// Native subgroup width range for this opened device, when exposed.
    /// This does not imply that every shader stage or operation uses subgroups.
    pub subgroup_widths: Option<SubgroupWidths>,
    /// Exact cooperative-matrix tuples accepted by the backend's native API.
    pub cooperative_matrix_configurations: Vec<CooperativeMatrixConfiguration>,
    /// Aggregate summary: true exactly when at least one scope-specific debug
    /// marker capability below is true. Core derives this bit when opening the
    /// device; backend initialization should set the scope facts.
    pub debug_markers: bool,
    pub debug_markers_encoder: bool,
    pub debug_markers_raster_pass: bool,
    pub debug_markers_compute_pass: bool,
    pub compute: bool,
    pub sparse: bool,
    pub mesh: bool,
    /// Acceleration structure sizing, creation, and build operations.
    pub acceleration_structure: bool,
    /// In-place updates of previously built acceleration structures.
    pub acceleration_structure_update: bool,
    /// Triangle geometry can read a buffer-backed affine transform when built.
    pub acceleration_geometry_transform: bool,
    /// A build can write the resulting acceleration structure size to a buffer.
    pub acceleration_build_size_output: bool,
    /// Compaction-enabled builds and their compacted size output are supported.
    pub acceleration_compaction_size: bool,
    /// Exact-sized acceleration compaction using backend-owned completed evidence.
    pub acceleration_compaction: bool,
    /// Fully bound, fixed-count binding arrays whose elements are
    /// acceleration structures. Runtime-sized and partially-bound arrays are
    /// separate capability domains.
    pub acceleration_structure_binding_arrays: bool,
    pub runtime_sized_acceleration_structure_binding_arrays: bool,
    pub partially_bound_acceleration_structure_binding_arrays: bool,
    pub ray: bool,
    pub ray_query: bool,
    /// WGSL/native fragment shaders may request early depth-test semantics.
    pub shader_early_depth_test: bool,
    /// Fragment shaders may read the rasterizer-assigned primitive index.
    pub shader_primitive_index: bool,
    /// Vertex shaders may read the draw index for a multi-draw operation.
    pub shader_draw_index: bool,
    /// A direct shader route may read standard perspective and linear
    /// barycentric coordinates (the Vulkan KHR/NV variants share this meaning).
    pub shader_barycentrics: bool,
    /// Standard barycentric SPIR-V route using `VK_KHR_fragment_shader_barycentric`.
    pub shader_barycentrics_khr: bool,
    /// Standard barycentric SPIR-V route using `VK_NV_fragment_shader_barycentric`.
    pub shader_barycentrics_nv: bool,
    /// A direct shader route may use the AMD explicit-vertex-parameter
    /// barycentric built-ins, whose center/centroid/sample modes are distinct.
    pub shader_barycentrics_amd: bool,
    /// Direct SPIR-V compute shaders may use `OpAtomicFAddEXT` on storage-buffer
    /// Float32 values. Other float atomic operations and address spaces are separate.
    pub shader_float32_atomic_buffer_add: bool,
    /// Direct SPIR-V compute `OpAtomicFAddEXT` on Float32 workgroup/shared memory.
    pub shader_float32_atomic_workgroup_add: bool,
    /// Direct SPIR-V compute `OpAtomicFMinEXT` on Float32 workgroup/shared memory.
    pub shader_float32_atomic_workgroup_min: bool,
    /// Direct SPIR-V compute `OpAtomicFMaxEXT` on Float32 workgroup/shared memory.
    pub shader_float32_atomic_workgroup_max: bool,
    /// Direct SPIR-V compute `OpAtomicFMinEXT` on Float32 storage-buffer memory.
    pub shader_float32_atomic_buffer_min: bool,
    /// Direct SPIR-V compute `OpAtomicFMaxEXT` on Float32 storage-buffer memory.
    pub shader_float32_atomic_buffer_max: bool,
    /// Direct SPIR-V compute shaders may use signed or unsigned Int64
    /// `OpAtomicIAdd` on storage-buffer memory. Other operations are separate.
    pub shader_int64_atomic_buffer_add: bool,
    /// Direct SPIR-V compute `OpAtomicSMin` on Int64 storage-buffer memory.
    pub shader_int64_atomic_buffer_min: bool,
    /// Direct SPIR-V compute `OpAtomicSMax` on Int64 storage-buffer memory.
    pub shader_int64_atomic_buffer_max: bool,
    /// Direct SPIR-V compute `OpAtomicUMin` on Int64 storage-buffer memory.
    pub shader_int64_atomic_buffer_umin: bool,
    /// Direct SPIR-V compute `OpAtomicUMax` on Int64 storage-buffer memory.
    pub shader_int64_atomic_buffer_umax: bool,
    /// Direct SPIR-V compute `OpAtomicXor` on Int64 storage-buffer memory.
    pub shader_int64_atomic_buffer_xor: bool,
    /// Direct SPIR-V compute `OpAtomicCompareExchange` on Int64 storage-buffer memory.
    pub shader_int64_atomic_buffer_compare_exchange: bool,
    /// Direct SPIR-V compute `OpAtomicExchange` on Int64 storage-buffer memory.
    pub shader_int64_atomic_buffer_exchange: bool,
    /// Direct SPIR-V compute `OpAtomicAnd` on Int64 storage-buffer memory.
    pub shader_int64_atomic_buffer_and: bool,
    /// Direct SPIR-V compute `OpAtomicOr` on Int64 storage-buffer memory.
    pub shader_int64_atomic_buffer_or: bool,
    /// Direct SPIR-V compute `OpAtomicIAdd` on Int64 workgroup/shared memory.
    pub shader_int64_atomic_workgroup_add: bool,
    /// Direct SPIR-V compute `OpAtomicSMin` on Int64 workgroup/shared memory.
    pub shader_int64_atomic_workgroup_min: bool,
    /// Direct SPIR-V compute `OpAtomicSMax` on Int64 workgroup/shared memory.
    pub shader_int64_atomic_workgroup_max: bool,
    /// Direct SPIR-V compute `OpAtomicXor` on Int64 workgroup/shared memory.
    pub shader_int64_atomic_workgroup_xor: bool,
    /// Direct SPIR-V compute `OpAtomicAnd` on Int64 workgroup/shared memory.
    pub shader_int64_atomic_workgroup_and: bool,
    /// Direct SPIR-V compute `OpAtomicOr` on Int64 workgroup/shared memory.
    pub shader_int64_atomic_workgroup_or: bool,
    /// Direct SPIR-V compute `OpAtomicUMin` on Int64 workgroup/shared memory.
    pub shader_int64_atomic_workgroup_umin: bool,
    /// Direct SPIR-V compute `OpAtomicUMax` on Int64 workgroup/shared memory.
    pub shader_int64_atomic_workgroup_umax: bool,
    /// Direct SPIR-V compute `OpAtomicExchange` on Int64 workgroup/shared memory.
    pub shader_int64_atomic_workgroup_exchange: bool,
    /// Direct SPIR-V compute `OpAtomicCompareExchange` on Int64 workgroup/shared memory.
    pub shader_int64_atomic_workgroup_compare_exchange: bool,
    /// Direct SPIR-V compute shaders may use `OpAtomicIAdd` on non-array,
    /// single-sample 2D R64Uint storage images.
    pub shader_int64_atomic_image_r64uint_add: bool,
    /// Direct SPIR-V compute shaders may use unsigned `OpAtomicUMin` on
    /// non-array, single-sample 2D R64Uint storage images.
    pub shader_int64_atomic_image_r64uint_umin: bool,
    /// Direct SPIR-V compute shaders may use unsigned `OpAtomicUMax` on
    /// non-array, single-sample 2D R64Uint storage images.
    pub shader_int64_atomic_image_r64uint_umax: bool,
    /// The backend has validated the direct-SPIR-V GLSL450 `StorageBuffer`
    /// `Coherent` decoration route for compute shaders. This is route
    /// admission, not a query of a native GPU feature or a synchronization
    /// guarantee; callers still provide the required synchronization.
    pub shader_memory_coherent_spirv: bool,
    /// The backend has validated the direct-SPIR-V GLSL450 `StorageBuffer`
    /// `Volatile` decoration route for compute shaders. This is route
    /// admission, not a query of a native GPU feature or a synchronization
    /// guarantee; callers still provide the required synchronization.
    pub shader_memory_volatile_spirv: bool,
    pub multi_queue: bool,
    pub indirect_draw: bool,
    /// Indirect records may contain a nonzero first-instance value, including
    /// its contribution to the shader instance index and instanced attributes.
    /// If false, callers must write zero to that argument field. Backends do
    /// not read GPU-produced records on the CPU to validate this precondition.
    pub indirect_first_instance: bool,
    pub multi_draw: bool,
    pub multi_draw_count: bool,
    /// Lowering facts when the backend has classified its supported routes.
    pub multi_draw_count_profile: Option<IndirectCountProfile>,
    /// Native batching of multiple mesh indirect records.
    pub mesh_multi_indirect: bool,
    /// Native GPU count-buffer execution for mesh indirect records.
    pub mesh_indirect_count: bool,
    pub push_constants: bool,
    pub storage_buffers: bool,
    pub storage_textures: bool,
    /// WGSL compute textureAtomicAdd on R32Uint storage textures.
    pub texture_atomic_r32uint_compute: bool,
    /// WGSL compute textureAtomicAdd on R32Sint storage textures.
    pub texture_atomic_r32sint_compute: bool,
    /// WGSL compute textureAtomicMin on R32Sint storage textures.
    pub texture_atomic_r32sint_min_compute: bool,
    /// WGSL compute textureAtomicMax on R32Sint storage textures.
    pub texture_atomic_r32sint_max_compute: bool,
    /// WGSL compute textureAtomicMin on R32Uint storage textures.
    pub texture_atomic_r32uint_min_compute: bool,
    /// WGSL compute textureAtomicMax on R32Uint storage textures.
    pub texture_atomic_r32uint_max_compute: bool,
    /// WGSL compute textureAtomicXor on R32Uint storage textures.
    pub texture_atomic_r32uint_xor_compute: bool,
    /// WGSL compute textureAtomicAnd on R32Uint storage textures.
    pub texture_atomic_r32uint_and_compute: bool,
    /// WGSL compute textureAtomicOr on R32Uint storage textures.
    pub texture_atomic_r32uint_or_compute: bool,
    pub binding_arrays: bool,
    /// A binding array whose count is chosen at bind-group creation time.
    /// This is deliberately separate from fixed-size binding arrays: shader
    /// indexing examples require both the native descriptor range and its
    /// variable-count validation path.
    pub runtime_sized_binding_arrays: bool,
    /// Compatibility aggregate for non-uniform sampled-texture and
    /// storage-buffer indexing. New admission uses the descriptor-kind facts
    /// below. Backends must not set this merely because they support arrays.
    pub nonuniform_sampled_texture_and_storage_buffer_indexing: bool,
    /// Non-uniform indexing capabilities by descriptor kind. The combined
    /// field above remains a compatibility summary; admission uses these facts.
    pub nonuniform_uniform_buffer_indexing: bool,
    pub nonuniform_storage_buffer_indexing: bool,
    pub nonuniform_sampled_texture_indexing: bool,
    /// Non-uniform indexing of storage texture binding arrays. This is
    /// separate because native APIs expose it independently from sampled
    /// texture and storage buffer indexing.
    pub nonuniform_storage_texture_indexing: bool,
    pub nonuniform_sampler_indexing: bool,
    pub nonuniform_acceleration_structure_indexing: bool,
    pub timestamp: bool,
    /// Timestamp queries can be written at an explicit point during a raster pass.
    pub timestamp_raster_pass: bool,
    /// Timestamp queries can be written at an explicit point during a compute pass.
    pub timestamp_compute_pass: bool,
    pub occlusion: bool,
    pub pipeline_statistics: bool,
    pub multiview: bool,
    /// Mesh pipeline creation accepts an explicit view count.
    pub mesh_multiview: bool,
    /// Vertex and fragment shaders can read the native view index.
    pub shader_view_index: bool,
    pub placed_memory: bool,
    pub external_memory: bool,
    pub secondary_commands: bool,
    pub native_interop: bool,
    /// Explicit allocator/native memory snapshot domain is available.
    pub memory_observation: bool,
    /// Dynamic raster depth-bias commands are supported.
    pub dynamic_depth_bias: bool,
    /// Dynamic depth-bias clamp values are supported.
    pub dynamic_depth_bias_clamp: bool,
    /// The raster pass may set different stencil references for the two faces.
    pub stencil_reference_per_face: bool,
    /// Raster and mesh pipelines may have independent front and back stencil masks.
    pub stencil_masks_per_face: bool,
    /// Clears selected color/depth/stencil attachments inside an active pass.
    pub rect_attachment_clear: bool,
    /// Conservative filled-triangle rasterization. `None` means the backend
    /// leaves admission to its pipeline path.
    pub conservative_raster: Option<bool>,
    /// Command encoders can populate backend-owned buffer upload staging directly.
    pub direct_buffer_upload: bool,
    /// Command encoders can populate backend-owned texture upload staging directly.
    pub direct_texture_upload: bool,
    /// Completion points support an explicit bounded host wait.
    pub completion_timeout: bool,
    /// Sampled color texture views can remap their shader-visible components.
    pub sampled_texture_component_mapping: bool,
    /// Texture views may restrict descriptor usage to a subset of the texture's usage.
    pub texture_view_usage: bool,
    /// A single depth slice of a 3D color texture can be a render attachment.
    pub texture_3d_attachment_slice: bool,
    /// Explicit color buffer/texture copies permit offsets and row pitches
    /// aligned to one texel or compressed block.
    pub texel_aligned_color_copies: bool,
    pub lanes: Arc<[LaneInfo]>,
    pub limits: DeviceLimits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SubgroupWidths {
    pub min: u32,
    pub max: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceMemoryTopology {
    Separate,
    Unified { cache_coherent: bool },
}

#[derive(Clone, Copy, Default)]
pub struct DeviceLimits {
    pub max_bind_groups: u32,
    pub max_bindings_per_group: u32,
    /// Shared descriptor-table ceilings for one pipeline layout, when the
    /// backend exposes this model. Arrays consume their declared capacity.
    pub descriptor_table_limits: Option<DescriptorTableLimits>,
    /// Upper bound on dynamic uniform buffer slots in one pipeline layout.
    /// Mixed layouts can have tighter limits. Zero means unreported.
    pub max_dynamic_uniform_buffers_per_pipeline_layout: u32,
    /// Upper bound on dynamic storage buffer slots in one pipeline layout.
    /// Mixed layouts can have tighter limits. Zero means unreported.
    pub max_dynamic_storage_buffers_per_pipeline_layout: u32,
    /// Maximum number of color slots in a raster pass or pipeline target
    /// list. Empty slots count toward the limit.
    pub max_color_attachments: u32,
    /// Maximum extent along the active axis of a 1D texture.
    /// A zero value leaves the maximum to the descriptor support query.
    pub max_texture_dimension_1d: u32,
    pub max_texture_dimension_2d: u32,
    /// Maximum extent along any axis of a 3D texture.
    /// A zero value leaves the maximum to the descriptor support query.
    pub max_texture_dimension_3d: u32,
    pub max_texture_array_layers: u32,
    pub max_buffer_bytes: u64,
    pub max_vertex_buffers: u32,
    pub max_vertex_attributes: u32,
    /// Maximum byte stride for a vertex buffer. Zero leaves the limit to the
    /// backend's pipeline admission checks.
    pub max_vertex_buffer_array_stride: u32,
    /// Maximum workgroup count per axis of a dispatch. A zero axis leaves
    /// admission for that axis to the backend.
    pub max_compute_workgroups: [u32; 3],
    pub max_compute_workgroup_size: [u32; 3],
    /// Combined invocation limit across all axes of one compute workgroup.
    /// Zero means the backend has not published a separate limit.
    pub max_compute_invocations_per_workgroup: u32,
    /// Maximum compute workgroup storage in bytes. Zero means unreported.
    pub max_compute_workgroup_storage_size: u32,
    /// Upper bound on the end offset of a push constant range, in bytes.
    /// Zero delegates size admission to the backend.
    pub max_push_constant_bytes: u32,
    pub max_multiview_views: u32,
    /// Maximum number of mesh indirect records consumed by one multi/count call.
    /// Zero means mesh multi-indirect execution is unavailable.
    pub max_mesh_indirect_count: u32,
    /// Total outer workgroups accepted by a direct mesh dispatch.
    /// This covers the task stage when the pipeline has one. Zero is unreported.
    pub max_mesh_workgroup_total_count: u32,
    /// Uniform ceiling for each axis of an outer mesh dispatch. Zero is unreported.
    pub max_mesh_workgroups_per_dimension: u32,
    /// Shader declaration and payload limits for the mesh/task stages.
    pub mesh_shader_limits: Option<MeshShaderLimits>,
    /// Maximum total number of ray-generation invocations in one trace.
    /// Zero means unreported or unavailable.
    pub max_ray_dispatch_count: u32,
    /// Maximum declared ray trace recursion depth. Zero means unreported or unavailable.
    pub max_ray_recursion_depth: u32,
    /// Acceleration structure build input bounds. Zero means unreported or unavailable.
    pub max_blas_primitive_count: u32,
    pub max_blas_geometry_count: u32,
    pub max_tlas_instance_count: u32,
    pub uniform_offset_alignment: u64,
    /// Maximum range length of one uniform buffer binding, in bytes.
    /// Zero leaves this limit to backend descriptor admission.
    pub max_uniform_buffer_binding_size: u64,
    pub storage_offset_alignment: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MeshShaderLimits {
    pub max_mesh_workgroup_invocations: u32,
    pub max_task_workgroup_invocations: u32,
    pub max_mesh_output_vertices: u32,
    pub max_mesh_output_primitives: u32,
    /// Exclusive upper bound for shader-written render target array indices.
    /// The bound attachment view may contain fewer layers.
    pub max_mesh_output_layers: u32,
    pub max_task_payload_bytes: u32,
    pub max_mesh_output_bytes: u32,
    /// Combined payload and output budget when the backend uses a shared limit.
    pub max_task_payload_and_mesh_output_bytes: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorTableLimits {
    pub uniform_buffers: u32,
    /// Sampled textures, read-only storage buffers and acceleration structures.
    pub read_only_resources: u32,
    /// Storage textures and writable storage buffers.
    pub writable_resources: u32,
    pub samplers: u32,
    /// The three resource classes share this additional ceiling. Samplers and
    /// bindings represented by root descriptors are counted separately.
    pub total_resource_descriptors: u32,
}
