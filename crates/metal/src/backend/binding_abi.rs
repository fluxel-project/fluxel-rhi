//! Per-pipeline Metal resource slots shared by shaders and vertex input.
//!
//! Shader buffers start at zero and vertex buffers occupy the upper slots;
//! their combined count must fit Metal's per-stage buffer limit.

use fluxel_rhi_core::api;

pub(crate) const MAX_VERTEX_BUFFERS: usize = 31;
pub(crate) const MAX_VERTEX_ATTRIBUTES: u32 = 31;

/// Namespace and native index assigned to one shader-visible binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SlotKind {
    Buffer,
    Texture,
    Sampler,
    /// Naga emits resource arrays as a Metal argument buffer.
    ArgumentBuffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BindingSlot {
    pub kind: SlotKind,
    pub index: u8,
}

#[derive(Clone, Debug)]
pub(crate) struct StageBindingMap {
    pub entries: Vec<(u32, u32, BindingSlot)>,
    pub push_slot: Option<u8>,
    pub sizes_slot: Option<u8>,
    pub view_slot: Option<u8>,
    pub ray_metadata: Vec<(u32, u32, u8)>,
}

impl StageBindingMap {
    pub fn slot(&self, group: u32, binding: u32) -> Option<BindingSlot> {
        self.entries
            .iter()
            .find_map(|&(g, b, slot)| (g == group && b == binding).then_some(slot))
    }
}

/// Allocate per-stage slots, reserving only native resources actually required
/// by that stage. Vertex input shares the buffer namespace with shader buffers.
pub(crate) fn stage_map(
    groups: &[super::bindings::MetalBindGroupLayout],
    ranges: &[api::PushConstantRange],
    stage: api::ShaderStage,
    vertex_buffers: usize,
    needs_sizes: bool,
    needs_view: bool,
    ray_metadata_bindings: &[(u32, u32)],
    texture_limit: usize,
) -> api::Result<StageBindingMap> {
    let mut buffers = 0usize;
    let mut textures = 0usize;
    let mut samplers = 0usize;
    let buffer_limit = 31usize
        .checked_sub(vertex_buffers)
        .ok_or(api::Error::InvalidRange)?;
    let mut result = StageBindingMap {
        entries: Vec::new(),
        push_slot: None,
        sizes_slot: None,
        view_slot: None,
        ray_metadata: Vec::new(),
    };
    for (group, layout) in groups.iter().enumerate() {
        for entry in &layout.entries {
            if !entry.visibility.contains(&stage) {
                continue;
            }
            let kind = if entry.count != api::BindingCount::One {
                SlotKind::ArgumentBuffer
            } else {
                match entry.ty {
                    api::BindingType::UniformBuffer { .. }
                    | api::BindingType::StorageBuffer { .. }
                    | api::BindingType::AccelerationStructure => SlotKind::Buffer,
                    api::BindingType::SampledTexture { .. }
                    | api::BindingType::StorageTexture { .. } => SlotKind::Texture,
                    api::BindingType::Sampler { .. } => SlotKind::Sampler,
                    api::BindingType::ExternalTexture => {
                        return Err(api::Error::Unsupported("Metal external texture binding"));
                    }
                }
            };
            let counter = match kind {
                SlotKind::Buffer | SlotKind::ArgumentBuffer => &mut buffers,
                SlotKind::Texture => &mut textures,
                SlotKind::Sampler => &mut samplers,
            };
            let index = u8::try_from(*counter).map_err(|_| api::Error::InvalidRange)?;
            *counter += 1;
            result.entries.push((
                u32::try_from(group).map_err(|_| api::Error::InvalidRange)?,
                entry.binding,
                BindingSlot { kind, index },
            ));
        }
    }
    if ranges.iter().any(|r| r.stages.contains(&stage)) {
        result.push_slot = Some(u8::try_from(buffers).map_err(|_| api::Error::InvalidRange)?);
        buffers += 1;
    }
    if needs_sizes {
        result.sizes_slot = Some(u8::try_from(buffers).map_err(|_| api::Error::InvalidRange)?);
        buffers += 1;
    }
    if needs_view {
        result.view_slot = Some(u8::try_from(buffers).map_err(|_| api::Error::InvalidRange)?);
        buffers += 1;
    }
    if !ray_metadata_bindings.is_empty() {
        for (group, layout) in groups.iter().enumerate() {
            for entry in &layout.entries {
                if entry.visibility.contains(&stage)
                    && ray_metadata_bindings.contains(&(group as u32, entry.binding))
                    && matches!(entry.ty, api::BindingType::AccelerationStructure)
                {
                    result.ray_metadata.push((
                        u32::try_from(group).map_err(|_| api::Error::InvalidRange)?,
                        entry.binding,
                        u8::try_from(buffers).map_err(|_| api::Error::InvalidRange)?,
                    ));
                    buffers += 1;
                }
            }
        }
    }
    if buffers > buffer_limit || textures > texture_limit || samplers > 16 {
        return Err(api::Error::Unsupported(
            "Metal per-stage resource slot limit",
        ));
    }
    Ok(result)
}
