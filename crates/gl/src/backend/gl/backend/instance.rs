//! Adopted-context adapter enumeration and device capability discovery.
//!
//! This module reports the one real adapter represented by an adopted GL
//! context. It performs no command recording or resource allocation.

use std::{
    future::{Ready, ready},
    sync::Arc,
};

use fluxel_rhi_core::api::{
    self, AdapterInfo, DeviceCapabilities, DeviceId, DeviceRequest, Error, InstanceBackend,
    InstanceId, LaneClass, LaneId, LaneInfo, Result,
};

use super::{Gl, GlDevice, GlInstance, spi::native_error, types::GlDeviceState};

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
        use crate::backend::gl::api::{GlCapability, GlFamilyApi as _};
        let (limits, compute_storage, indirect_draw, occlusion) = state
            .with_native("read GL device limits", |provider| {
                let discovery = provider.discovery();
                let capabilities = discovery.capabilities();
                // A public compute route needs both a real dispatch probe and
                // a storage-image probe: example 06 cannot be useful with
                // only one half, and reporting it would make Core advertise a
                // pipeline it cannot bind.
                Ok((
                    discovery.limits(),
                    capabilities.supports(GlCapability::Compute)
                        && capabilities.supports(GlCapability::StorageImage),
                    capabilities.supports(GlCapability::IndirectDraw),
                    capabilities.supports(GlCapability::OcclusionQuery),
                ))
            })
            .map_err(native_error)?;
        return Ok(DeviceCapabilities {
            lanes: Arc::from([LaneInfo {
                id: LaneId { device, ordinal: 0 },
                class: LaneClass::Graphics,
                routes: Vec::new(),
            }]),
            compute: compute_storage,
            storage_textures: compute_storage,
            indirect_draw,
            occlusion,
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
                // GL does not expose a portable maximum buffer allocation.
                // The direct resource route validates sizes against GLsizei.
                max_buffer_bytes: i32::MAX as u64,
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
                max_push_constant_bytes: 0,
                max_multiview_views: 0,
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
            return Ok(DeviceCapabilities {
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
                    max_push_constant_bytes: 0,
                    max_multiview_views: 0,
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
