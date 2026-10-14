//! Capability publication for one discovered GL context.
//!
//! Discovery records profile, extension and entry-point evidence in this
//! snapshot. This module publishes only operations with both that evidence and
//! an implemented lowerer. It does not probe drivers, allocate resources, or
//! infer RenderGraph resource states.

use std::sync::Arc;

use fluxel_rhi_core::api::{
    DeviceCapabilities, DeviceId, DeviceLimits, LaneClass, LaneId, LaneInfo,
};

use super::{
    api::{GlFormat, GlLimits},
    facts::{GlFeature, GlFeatureProbe},
    profile::GlProfile,
};

/// Native-format evidence retained for resource creation admission.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GlFormatEvidence {
    pub(crate) format: GlFormat,
    pub(crate) filterable: bool,
    pub(crate) renderable: bool,
    pub(crate) blendable: bool,
}

/// A lowering implementation boundary, deliberately separate from driver facts.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GlLoweringClosure {
    pub(crate) buffers: bool,
    pub(crate) textures: bool,
    pub(crate) buffer_copy: bool,
    pub(crate) texture_copy: bool,
    pub(crate) raster: bool,
    pub(crate) compute: bool,
    pub(crate) raster_indirect: bool,
    pub(crate) occlusion_query: bool,
}

/// Immutable discovery result for one backend-private context generation.
#[derive(Clone, Debug)]
pub(crate) struct GlCapabilitySnapshot {
    pub(crate) profile: GlProfile,
    pub(crate) feature_probe: GlFeatureProbe,
    pub(crate) limits: GlLimits,
    pub(crate) maximum_buffer_size: u64,
    pub(crate) formats: Vec<GlFormatEvidence>,
    pub(crate) lowering: GlLoweringClosure,
}

impl GlCapabilitySnapshot {
    /// Publishes current Core capabilities for this exact opened device.
    ///
    /// A feature bit remains false until the selected profile route, its
    /// required functions, and the backend lowerer all agree. GL has one
    /// graphics lane; additional native queues are never fabricated here.
    pub(crate) fn into_capabilities(self, device: DeviceId) -> DeviceCapabilities {
        let indirect =
            self.lowering.raster_indirect && self.feature_probe.supports(GlFeature::IndirectDraw);
        let compute = self.lowering.compute && self.feature_probe.supports(GlFeature::Compute);
        let occlusion =
            self.lowering.occlusion_query && self.feature_probe.supports(GlFeature::OcclusionQuery);
        DeviceCapabilities {
            compute,
            indirect_draw: indirect,
            occlusion,
            storage_buffers: compute && self.limits.supports_storage_buffers(),
            storage_textures: compute && self.limits.supports_storage_images(),
            lanes: Arc::from([LaneInfo {
                id: LaneId { device, ordinal: 0 },
                class: LaneClass::Graphics,
                routes: Vec::new(),
            }]),
            limits: DeviceLimits {
                max_color_attachments: self.limits.max_color_attachments,
                max_texture_dimension_2d: self.limits.max_texture_size,
                max_texture_array_layers: self.limits.max_array_texture_layers,
                max_buffer_bytes: self.maximum_buffer_size,
                max_vertex_buffers: self.limits.max_vertex_attributes,
                max_vertex_attributes: self.limits.max_vertex_attributes,
                max_compute_workgroups: self.limits.max_compute_work_group_count,
                max_compute_workgroup_size: self.limits.max_compute_work_group_size,
                uniform_offset_alignment: self.limits.uniform_buffer_offset_alignment,
                storage_offset_alignment: self.limits.storage_buffer_offset_alignment,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    /// Internal format admission is exact: a represented format is not
    /// automatically filterable or renderable merely because GL names it.
    pub(crate) fn format(&self, format: GlFormat) -> Option<GlFormatEvidence> {
        self.formats
            .iter()
            .copied()
            .find(|known| known.format == format)
    }
}
