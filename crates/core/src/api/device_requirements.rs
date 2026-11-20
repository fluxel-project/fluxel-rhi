//! Feature and limit requirements for opening a device for a known workload.
//!
//! Requirements qualify an opened device. They do not enable unavailable
//! native operations or relax descriptor-specific admission checks.

use super::{DeviceCapabilities, DeviceLimits, DeviceRequest, Error, Result};

macro_rules! device_features {
    ($($feature:ident => $field:ident, $name:literal);+ $(;)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum DeviceFeature {
            $($feature),+
        }

        impl DeviceFeature {
            /// Name accepted by `DeviceRequest::required_features`.
            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$feature => $name),+
                }
            }
        }

        impl DeviceCapabilities {
            pub fn supports(&self, feature: DeviceFeature) -> bool {
                match feature {
                    $(DeviceFeature::$feature => self.$field),+
                }
            }

            #[doc(hidden)]
            pub fn require_named_features(&self, features: &[&'static str]) -> Result<()> {
                for &name in features {
                    let supported = match name {
                        $($name => self.supports(DeviceFeature::$feature)),+,
                        _ => false,
                    };
                    if !supported {
                        return Err(Error::Unsupported(name));
                    }
                }
                Ok(())
            }
        }
    };
}

device_features! {
    Compute => compute, "compute";
    ShaderF16 => shader_f16, "shader-f16";
    ShaderI16 => shader_i16, "shader-i16";
    ShaderF64 => shader_f64, "shader-f64";
    ShaderInt64 => shader_int64, "shader-int64";
    Sparse => sparse, "sparse";
    Mesh => mesh, "mesh";
    MeshMultiIndirect => mesh_multi_indirect, "mesh-multi-indirect";
    MeshIndirectCount => mesh_indirect_count, "mesh-indirect-count";
    AccelerationStructure => acceleration_structure, "acceleration-structure";
    RayPipeline => ray, "ray-pipeline";
    RayQuery => ray_query, "ray-query";
    MultiQueue => multi_queue, "multi-queue";
    IndirectDraw => indirect_draw, "indirect-draw";
    IndirectFirstInstance => indirect_first_instance, "indirect-first-instance";
    MultiDraw => multi_draw, "multi-draw";
    MultiDrawCount => multi_draw_count, "multi-draw-count";
    PushConstants => push_constants, "push-constants";
    StorageBuffers => storage_buffers, "storage-buffers";
    GpuUploadBuffers => gpu_upload_buffers, "gpu-upload-buffers";
    StorageTextures => storage_textures, "storage-textures";
    BindingArrays => binding_arrays, "binding-arrays";
    RuntimeSizedBindingArrays => runtime_sized_binding_arrays, "runtime-sized-binding-arrays";
    Timestamp => timestamp, "timestamp";
    Occlusion => occlusion, "occlusion";
    PipelineStatistics => pipeline_statistics, "pipeline-statistics";
    Multiview => multiview, "multiview";
    MeshMultiview => mesh_multiview, "mesh-multiview";
    PlacedMemory => placed_memory, "placed-memory";
    ExplicitHeapResidency => explicit_heap_residency, "explicit-heap-residency";
    ExternalMemory => external_memory, "external-memory";
    SecondaryCommands => secondary_commands, "secondary-commands";
    DynamicDepthBias => dynamic_depth_bias, "dynamic-depth-bias";
    RectAttachmentClear => rect_attachment_clear, "rect-attachment-clear";
}

impl DeviceRequest {
    /// Adds a required feature to the request. The backend may reject requests
    /// that it cannot qualify before returning the device.
    pub fn require_feature(mut self, feature: DeviceFeature) -> Self {
        self.required_features.push(feature.name());
        self
    }
}

/// Minimum capacity required by a caller. Each value uses the units of the
/// corresponding upper bound in `DeviceLimits`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceCapacity {
    BufferBytes(u64),
    TextureDimension1d(u32),
    TextureDimension2d(u32),
    TextureDimension3d(u32),
    TextureArrayLayers(u32),
    BindGroups(u32),
    BindingsPerGroup(u32),
    DynamicUniformBuffersPerPipelineLayout(u32),
    DynamicStorageBuffersPerPipelineLayout(u32),
    UniformBufferBindingSize(u64),
    ColorAttachments(u32),
    VertexBuffers(u32),
    VertexAttributes(u32),
    VertexBufferArrayStride(u32),
    ComputeWorkgroups([u32; 3]),
    ComputeWorkgroupSize([u32; 3]),
    ComputeInvocationsPerWorkgroup(u32),
    ComputeWorkgroupStorageSize(u32),
    PushConstantBytes(u32),
    MultiviewViews(u32),
    MeshIndirectCount(u32),
    MeshWorkgroupTotalCount(u32),
    MeshWorkgroupsPerDimension(u32),
    RayDispatchCount(u32),
    RayRecursionDepth(u32),
    BlasPrimitiveCount(u32),
    BlasGeometryCount(u32),
    TlasInstanceCount(u32),
}

impl DeviceCapacity {
    pub fn is_met_by(self, limits: &DeviceLimits) -> bool {
        match self {
            Self::BufferBytes(min) => limits.max_buffer_bytes >= min,
            Self::TextureDimension1d(min) => limits.max_texture_dimension_1d >= min,
            Self::TextureDimension2d(min) => limits.max_texture_dimension_2d >= min,
            Self::TextureDimension3d(min) => limits.max_texture_dimension_3d >= min,
            Self::TextureArrayLayers(min) => limits.max_texture_array_layers >= min,
            Self::BindGroups(min) => limits.max_bind_groups >= min,
            Self::BindingsPerGroup(min) => limits.max_bindings_per_group >= min,
            Self::DynamicUniformBuffersPerPipelineLayout(min) => {
                limits.max_dynamic_uniform_buffers_per_pipeline_layout >= min
            }
            Self::DynamicStorageBuffersPerPipelineLayout(min) => {
                limits.max_dynamic_storage_buffers_per_pipeline_layout >= min
            }
            Self::UniformBufferBindingSize(min) => limits.max_uniform_buffer_binding_size >= min,
            Self::ColorAttachments(min) => limits.max_color_attachments >= min,
            Self::VertexBuffers(min) => limits.max_vertex_buffers >= min,
            Self::VertexAttributes(min) => limits.max_vertex_attributes >= min,
            Self::VertexBufferArrayStride(min) => limits.max_vertex_buffer_array_stride >= min,
            Self::ComputeWorkgroups(min) => limits
                .max_compute_workgroups
                .into_iter()
                .zip(min)
                .all(|(limit, min)| limit >= min),
            Self::ComputeWorkgroupSize(min) => limits
                .max_compute_workgroup_size
                .into_iter()
                .zip(min)
                .all(|(limit, min)| limit >= min),
            Self::ComputeInvocationsPerWorkgroup(min) => {
                limits.max_compute_invocations_per_workgroup >= min
            }
            Self::ComputeWorkgroupStorageSize(min) => {
                limits.max_compute_workgroup_storage_size >= min
            }
            Self::PushConstantBytes(min) => limits.max_push_constant_bytes >= min,
            Self::MultiviewViews(min) => limits.max_multiview_views >= min,
            Self::MeshIndirectCount(min) => limits.max_mesh_indirect_count >= min,
            Self::MeshWorkgroupTotalCount(min) => limits.max_mesh_workgroup_total_count >= min,
            Self::MeshWorkgroupsPerDimension(min) => {
                limits.max_mesh_workgroups_per_dimension >= min
            }
            Self::RayDispatchCount(min) => limits.max_ray_dispatch_count >= min,
            Self::RayRecursionDepth(min) => limits.max_ray_recursion_depth >= min,
            Self::BlasPrimitiveCount(min) => limits.max_blas_primitive_count >= min,
            Self::BlasGeometryCount(min) => limits.max_blas_geometry_count >= min,
            Self::TlasInstanceCount(min) => limits.max_tlas_instance_count >= min,
        }
    }
}

impl DeviceLimits {
    #[doc(hidden)]
    pub fn require_capacities(&self, minimum: &[DeviceCapacity]) -> Result<()> {
        if minimum.iter().any(|capacity| !capacity.is_met_by(self)) {
            return Err(Error::Unsupported("requested device capacity"));
        }
        Ok(())
    }
}
