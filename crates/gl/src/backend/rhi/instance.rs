//! Adopted-context adapter enumeration and device capability discovery.
//!
//! This module reports the one real adapter represented by an adopted GL
//! context. It performs no command recording or resource allocation.

use std::{
    future::{Ready, ready},
    sync::Arc,
};

use fluxel_rhi_core::api::{
    self, AdapterInfo, DeviceCapabilities, DeviceId, DeviceRequest, Error, InstanceId, LaneClass,
    LaneId, LaneInfo, Result,
};
use fluxel_rhi_core::backend::InstanceBackend;

use super::{Gl, GlDevice, GlInstance, spi::native_error, types::GlDeviceState};

pub(super) fn push_limit(state: &std::rc::Rc<GlDeviceState>) -> Result<u32> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::api::GlFamilyApi as _;
        return state
            .with_native("query GL push constant limit", |provider| {
                let discovery = provider.discovery();
                Ok(discovery
                    .limits()
                    .push_constant_bytes(discovery.capabilities().supports_compute()))
            })
            .map_err(native_error);
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        return Ok(state
            .owner
            .discovery_snapshot()
            .limits()
            .push_constant_bytes(false));
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = state;
        Ok(0)
    }
}

pub(super) fn multiview_limit(state: &std::rc::Rc<GlDeviceState>) -> Result<u32> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::api::{GlCapability, GlFamilyApi as _};
        return state
            .with_native("query multiview limits", |provider| {
                let discovery = provider.discovery();
                Ok(
                    if discovery.capabilities().supports(GlCapability::Multiview) {
                        discovery.max_multiview_view_count().min(32)
                    } else {
                        0
                    },
                )
            })
            .map_err(native_error);
    }
    #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
    {
        let discovery = state.owner.discovery_snapshot();
        return Ok(
            if discovery
                .capabilities()
                .supports(crate::backend::api::GlCapability::Multiview)
            {
                discovery.max_multiview_view_count().min(32)
            } else {
                0
            },
        );
    }
    #[cfg(not(any(
        all(windows, feature = "native-gl-wgl"),
        feature = "native-gles-egl",
        all(target_arch = "wasm32", feature = "webgl2")
    )))]
    {
        let _ = state;
        Ok(0)
    }
}

impl InstanceBackend for Gl {
    type NativeInstance = GlInstance;
    type OpenFuture<'a> = Ready<Result<(GlDevice, DeviceCapabilities)>>;

    fn enumerate(instance: &GlInstance, scope: InstanceId) -> Result<Vec<AdapterInfo>> {
        let name = instance
            .0
            .adapter_name()
            .map_err(|error| Error::Backend(format!("{error:?}")))?;
        Ok(vec![AdapterInfo {
            id: api::AdapterId {
                instance: scope,
                ordinal: 0,
            },
            name,
        }])
    }

    fn open<'a>(
        instance: &'a GlInstance,
        device: DeviceId,
        request: &'a DeviceRequest,
    ) -> Self::OpenFuture<'a> {
        if request.adapter.ordinal != 0 || !request.required_features.is_empty() {
            return ready(Err(Error::Unsupported("requested GL device features")));
        }
        let caps = gl_capabilities(&instance.0, device);
        ready(caps.map(|caps| (GlDevice(instance.0.clone()), caps)))
    }
}

fn gl_capabilities(
    state: &std::rc::Rc<GlDeviceState>,
    device: DeviceId,
) -> Result<DeviceCapabilities> {
    #[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
    {
        use crate::backend::api::{GlCapability, GlFamilyApi as _};
        let (
            limits,
            compute_storage,
            indirect_draw,
            indirect_count,
            occlusion,
            storage_buffers,
            storage_images,
            native_binding_arrays,
            wide_buffer_abi,
            debug_markers,
        ) = state
            .with_native("read GL device limits", |provider| {
                let discovery = provider.discovery();
                let capabilities = discovery.capabilities();
                // Dispatch and storage-image probes describe independent
                // capabilities; individual bindings retain their own gates.
                Ok((
                    discovery.limits(),
                    capabilities.supports_compute(),
                    capabilities.supports(GlCapability::IndirectDraw),
                    provider.supports_indirect_count(),
                    capabilities.supports(GlCapability::OcclusionQuery),
                    capabilities.supports(GlCapability::StorageBuffer),
                    capabilities.supports(GlCapability::StorageImage),
                    matches!(provider.profile(), crate::backend::api::GlFamilyProfile::Desktop { major, .. } if major >= 4) || matches!(provider.profile(), crate::backend::api::GlFamilyProfile::Embedded { major, .. } if major >= 3),
                    provider.supports_wide_buffer_abi(),
                    provider.debug_markers_supported(),
                ))
            })
            .map_err(native_error)?;
        let native_base_instance = state.with_native("read GL base instance support", |provider| {
            Ok(matches!(provider.profile(), crate::backend::api::GlFamilyProfile::Desktop { major, minor } if major > 4 || major == 4 && minor >= 2)
                || provider.discovery().extensions().raw_reported_names().any(|name| name == "GL_ARB_base_instance"))
        }).map_err(native_error)?;
        let multiview_views = multiview_limit(state)?;
        return Ok(DeviceCapabilities {
            debug_markers,
            debug_markers_encoder: debug_markers,
            debug_markers_raster_pass: debug_markers,
            debug_markers_compute_pass: debug_markers,
            lanes: Arc::from([LaneInfo {
                id: LaneId { device, ordinal: 0 },
                class: LaneClass::Graphics,
                routes: Vec::new(),
            }]),
            compute: compute_storage,
            native_interop: true,
            // Native runtime arrays specialize to the layout capacity. Each
            // candidate uses ordinary native slots; pipeline limits still
            // bound resource counts and texture/sampler pair expansion.
            binding_arrays: true,
            runtime_sized_binding_arrays: native_binding_arrays,
            nonuniform_sampled_texture_and_storage_buffer_indexing: native_binding_arrays
                && storage_buffers,
            nonuniform_uniform_buffer_indexing: native_binding_arrays && storage_buffers,
            nonuniform_storage_buffer_indexing: native_binding_arrays && storage_buffers,
            nonuniform_sampled_texture_indexing: native_binding_arrays && storage_buffers,
            nonuniform_sampler_indexing: native_binding_arrays && storage_buffers,
            storage_textures: storage_images,
            storage_buffers,
            indirect_draw,
            indirect_first_instance: indirect_draw
                && native_base_instance
                && super::shader::draw_parameters(&GlDevice(state.clone()))?
                    != super::shader::DrawParameters::None,
            multi_draw: indirect_draw,
            multi_draw_count: indirect_count,
            push_constants: limits.push_constant_bytes(compute_storage) != 0,
            multiview: multiview_views > 0,
            occlusion,
            timestamp: super::query::query_supported(
                &GlDevice(state.clone()),
                api::QueryKind::Timestamp,
            ),
            pipeline_statistics: super::query::query_supported(
                &GlDevice(state.clone()),
                api::QueryKind::PipelineStatistics,
            ),
            limits: api::DeviceLimits {
                // GL has no native bind-group object: logical groups are
                // flattened to UBO points and texture units at pipeline link.
                // Four groups preserve the WebGPU baseline without claiming a
                // driver limit that GL does not expose.
                max_bind_groups: 4,
                max_bindings_per_group: limits
                    .max_uniform_buffer_bindings
                    .min(limits.max_combined_texture_image_units),
                max_color_attachments: limits.max_color_attachments,
                max_texture_dimension_2d: limits.max_texture_size,
                max_texture_array_layers: limits.max_array_texture_layers,
                // GL does not expose a portable maximum allocation. Native
                // `GLsizeiptr`/`GLintptr` are pointer-sized, so publish the
                // ABI addressing ceiling; a real allocation may still report
                // GL_OUT_OF_MEMORY. WebGL keeps its separate JS-facing bound.
                max_buffer_bytes:
                    crate::backend::native::buffer_abi::advertised_native_max_buffer_bytes(
                        wide_buffer_abi,
                    ),
                // Vertex attributes may source distinct buffers through one
                // VAO. GL exposes an attribute-count limit, not a separate
                // binding-slot count, therefore this is the proven ceiling.
                max_vertex_buffers: limits.max_vertex_attributes,
                max_vertex_attributes: limits.max_vertex_attributes,
                max_compute_workgroups: compute_storage
                    .then_some(limits.max_compute_work_group_count)
                    .unwrap_or([0; 3]),
                max_compute_workgroup_size: compute_storage
                    .then_some(limits.max_compute_work_group_size)
                    .unwrap_or([0; 3]),
                max_push_constant_bytes: limits.push_constant_bytes(compute_storage),
                max_multiview_views: multiview_views,
                max_mesh_indirect_count: 0,
                uniform_offset_alignment: limits.uniform_buffer_offset_alignment,
                storage_offset_alignment: limits.storage_buffer_offset_alignment,
            },
            ..Default::default()
        });
    }
    #[cfg(not(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl")))]
    {
        #[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
        {
            let limits = state.owner.discovery_snapshot().limits();
            let multiview_views = multiview_limit(state)?;
            return Ok(DeviceCapabilities {
                push_constants: limits.push_constant_bytes(false) != 0,
                // WebGL2 has no unsized descriptor-array primitive, but the
                // Core contract is explicitly bounded: the GL lowering
                // specializes each RuntimeSized declaration to its layout's
                // max_count and assigns ordinary WebGL texture/UBO slots.
                // Layout creation rejects a capacity which cannot fit this
                // context before shader specialization can expand it.
                //
                // Do not expose the combined nonuniform feature here. Its
                // public contract also includes SSBO indexing, which WebGL2
                // does not provide; ESSL opaque-array indexing consequently
                // remains limited to the dynamically-uniform cases accepted
                // by WebGL2.
                binding_arrays: true,
                runtime_sized_binding_arrays: true,
                multiview: multiview_views > 0,
                occlusion: true,
                timestamp: super::query::query_supported(
                    &GlDevice(state.clone()),
                    api::QueryKind::Timestamp,
                ),
                lanes: Arc::from([LaneInfo {
                    id: LaneId { device, ordinal: 0 },
                    class: LaneClass::Graphics,
                    routes: Vec::new(),
                }]),
                limits: api::DeviceLimits {
                    // Logical groups are flattened to WebGL2 UBO points and
                    // texture units at link time. Four groups preserve the
                    // WebGPU baseline; native slot capacity stays driven by
                    // the limits queried from this exact browser context.
                    max_bind_groups: 4,
                    max_bindings_per_group: limits
                        .max_uniform_buffer_bindings
                        .min(limits.max_combined_texture_image_units),
                    max_color_attachments: limits.max_color_attachments,
                    max_texture_dimension_2d: limits.max_texture_size,
                    max_texture_array_layers: limits.max_array_texture_layers,
                    max_buffer_bytes: i32::MAX as u64,
                    max_vertex_buffers: limits.max_vertex_attributes,
                    max_vertex_attributes: limits.max_vertex_attributes,
                    max_compute_workgroups: [0; 3],
                    max_compute_workgroup_size: [0; 3],
                    max_push_constant_bytes: limits.push_constant_bytes(false),
                    max_multiview_views: multiview_views,
                    max_mesh_indirect_count: 0,
                    uniform_offset_alignment: limits.uniform_buffer_offset_alignment,
                    storage_offset_alignment: 0,
                },
                ..Default::default()
            });
        }
        #[cfg(not(all(target_arch = "wasm32", feature = "webgl2")))]
        {
            let _ = (state, device);
            Err(Error::Unsupported("GL platform context"))
        }
    }
}
