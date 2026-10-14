//! Platform-neutral resource descriptions and validation.
//!
//! This module owns the descriptors for buffers, textures, texture views and
//! renderbuffers, the rules a descriptor must satisfy before any provider
//! allocates storage for it, and the range and subresource arithmetic those
//! rules are expressed in. It does not own allocation or object lifetime (that
//! is the provider) and it does not own the context facts a rule is checked
//! against: a provider supplies its discovered limits and format evidence, and
//! this module states the rule and the structured reason a violated one
//! produces.
//!
//! # Compressed mip chains
//!
//! A recorded contract decision, written here because the rule this module does
//! *not* enforce has to be a statement rather than silence.
//!
//! What is enforced: a compressed upload must define exactly one complete 2D
//! mip, and the byte count it supplies must equal that mip's exact encoded size
//! (`GlCompressedFormatInfo::checked_encoded_size`) rather than anything the
//! client asserts. Compressed storage stays undefined until something defines
//! it, so that rule is what makes a compressed upload mean anything; it lives
//! with the recorder's upload path and it is closed.
//!
//! What is deliberately not enforced: nothing requires every mip of a
//! `mip_level_count > 1` chain to be defined. A descriptor may therefore
//! allocate a whole chain while the caller fills in only some of its levels, and
//! no layer rejects that -- not `GlTextureDesc::validate`, which checks that the
//! requested count is legal for the extent and the format but never that the
//! chain is complete, and not any upload, which sees one subresource and cannot
//! see the chain it belongs to. A chain can stay permanently partial without the
//! descriptor ever being rejected.
//!
//! Why a partial chain is the caller's problem at this layer: completeness is a
//! distribution-of-uploads fact about an allocation's history, not a property of
//! any single command, so enforcing it needs per-texture defined-mip state.
//! Nothing needs that state yet -- reading a texture is a Layer 2/3 concern, and
//! this layer's contract is the command vocabulary plus per-command validation.
//! The tracking could not live in this module in any case: a descriptor here is
//! a value that is copied, compared and hashed, not a record of what has
//! happened to an allocation since it was created. Adding lifetime-scoped state
//! now would be an API built for a future need rather than for a current one.
//!
//! What would have to change to revisit it: a real consumer that needs a
//! complete chain before a read -- a Layer 2/3 path that samples a texture whose
//! levels the caller may only partly have uploaded. Chain completeness would
//! then be tracked where the lifetime already lives (the provider's object
//! store, keyed by texture identity and invalidated with the context epoch),
//! no descriptor field would change, and the rejection would belong at the first
//! command that reads the texture rather than at allocation, because allocation
//! is not the point at which a partial chain becomes wrong.

use super::{BufferId, GlFormat, RenderbufferId, TextureId};

/// Buffer operations permitted for a resource.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct GlBufferUsage(u32);

impl GlBufferUsage {
    pub(crate) const COPY_SOURCE: Self = Self(1 << 0);
    pub(crate) const COPY_DESTINATION: Self = Self(1 << 1);
    pub(crate) const VERTEX: Self = Self(1 << 2);
    pub(crate) const INDEX: Self = Self(1 << 3);
    pub(crate) const UNIFORM: Self = Self(1 << 4);
    pub(crate) const STORAGE: Self = Self(1 << 5);
    pub(crate) const INDIRECT: Self = Self(1 << 6);
    pub(crate) const MAP_READ: Self = Self(1 << 7);
    pub(crate) const MAP_WRITE: Self = Self(1 << 8);
    pub(crate) const EMPTY: Self = Self(0);

    pub(crate) const fn contains(self, required: Self) -> bool {
        self.0 & required.0 == required.0
    }
    pub(crate) const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl core::ops::BitOr for GlBufferUsage {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// One binding role a buffer can be created for.
///
/// The roles are named after what the GPU does with the bytes. They exist so a
/// descriptor can be rejected *before* any driver object is created, and so the
/// rejection can say which two roles collided instead of reporting a generic
/// invalid descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlBufferRole {
    /// Attribute source.
    Vertex,
    /// Element source.
    Index,
    /// Uniform block source.
    Uniform,
    /// Shader storage block source, where the profile has one.
    Storage,
    /// Draw/dispatch parameter source, where the profile has one.
    Indirect,
    /// Source of a buffer-to-buffer or buffer-to-texture copy.
    CopySource,
    /// Destination of a buffer-to-buffer or texture-to-buffer copy.
    CopyDestination,
    /// CPU read mapping: a readback staging buffer.
    MapRead,
    /// CPU write mapping: an upload staging buffer.
    MapWrite,
}

impl GlBufferRole {
    /// Every role, in a fixed order so a rejection is deterministic.
    pub(crate) const ALL: [Self; 9] = [
        Self::MapRead,
        Self::MapWrite,
        Self::Vertex,
        Self::Index,
        Self::Uniform,
        Self::Storage,
        Self::Indirect,
        Self::CopySource,
        Self::CopyDestination,
    ];

    pub(crate) const fn usage(self) -> GlBufferUsage {
        match self {
            Self::Vertex => GlBufferUsage::VERTEX,
            Self::Index => GlBufferUsage::INDEX,
            Self::Uniform => GlBufferUsage::UNIFORM,
            Self::Storage => GlBufferUsage::STORAGE,
            Self::Indirect => GlBufferUsage::INDIRECT,
            Self::CopySource => GlBufferUsage::COPY_SOURCE,
            Self::CopyDestination => GlBufferUsage::COPY_DESTINATION,
            Self::MapRead => GlBufferUsage::MAP_READ,
            Self::MapWrite => GlBufferUsage::MAP_WRITE,
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Index => "index",
            Self::Uniform => "uniform",
            Self::Storage => "storage",
            Self::Indirect => "indirect",
            Self::CopySource => "copy-source",
            Self::CopyDestination => "copy-destination",
            Self::MapRead => "map-read",
            Self::MapWrite => "map-write",
        }
    }

    /// Whether the role describes a CPU mapping rather than a GPU use.
    const fn is_mapping(self) -> bool {
        matches!(self, Self::MapRead | Self::MapWrite)
    }

    /// Whether the GPU reads the buffer through a binding point as a resource.
    const fn is_gpu_resource(self) -> bool {
        matches!(
            self,
            Self::Vertex | Self::Index | Self::Uniform | Self::Storage | Self::Indirect
        )
    }
}

/// Whether two roles cannot be declared on one buffer.
///
/// GL stores an untyped byte range, so one buffer legitimately serves several
/// consumption roles at once: vertex, index, uniform, copy-source, and
/// copy-destination may be combined freely, and storage and indirect join them
/// wherever the profile has those domains. Rejecting those pairings here would
/// invent a restriction neither the GL family nor the WebGPU-aligned model has;
/// they are validated where they are used instead, by requiring the descriptor
/// to declare the role being bound.
///
/// Mapping is the exception, and the only reason this table exists. A mapped
/// buffer is a staging buffer: its contents are well defined only while the CPU
/// holds the mapping, and the GL family gives no coherence between a mapping
/// and a concurrent GPU read of the same bytes. So a mapping role conflicts
/// with the other mapping role and with every role that lets the GPU consume
/// the buffer as a resource. Copy roles stay legal next to a mapping -- that is
/// exactly the upload and readback staging pattern.
const fn roles_conflict(first: GlBufferRole, second: GlBufferRole) -> bool {
    if first.is_mapping() {
        second.is_mapping() || second.is_gpu_resource()
    } else if second.is_mapping() {
        first.is_gpu_resource()
    } else {
        false
    }
}

impl GlBufferUsage {
    /// The first conflicting role pair this usage set declares, if any.
    ///
    /// Roles are examined in [`GlBufferRole::ALL`] order, so the same usage set
    /// always reports the same pair.
    pub(crate) const fn conflicting_roles(self) -> Option<(GlBufferRole, GlBufferRole)> {
        let mut i = 0;
        while i < GlBufferRole::ALL.len() {
            let first = GlBufferRole::ALL[i];
            if self.contains(first.usage()) {
                let mut j = i + 1;
                while j < GlBufferRole::ALL.len() {
                    let second = GlBufferRole::ALL[j];
                    if self.contains(second.usage()) && roles_conflict(first, second) {
                        return Some((first, second));
                    }
                    j += 1;
                }
            }
            i += 1;
        }
        None
    }
}

/// Immutable creation facts for a buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlBufferDesc {
    pub size: u64,
    pub usage: GlBufferUsage,
}

impl GlBufferDesc {
    pub(crate) fn validate(self) -> Result<(), GlResourceValidationError> {
        if self.size == 0 {
            return Err(GlResourceValidationError::ZeroBufferSize);
        }
        if self.usage.is_empty() {
            return Err(GlResourceValidationError::EmptyBufferUsage);
        }
        if let Some((first, second)) = self.usage.conflicting_roles() {
            return Err(GlResourceValidationError::ConflictingBufferRoles { first, second });
        }
        Ok(())
    }
}

/// A byte range in one buffer.  `size` is deliberately never implicit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlBufferRange {
    pub buffer: BufferId,
    pub offset: u64,
    pub size: u64,
}

impl GlBufferRange {
    pub(crate) fn validate_for(self, desc: GlBufferDesc) -> Result<(), GlResourceValidationError> {
        if self.size == 0 {
            return Err(GlResourceValidationError::ZeroRangeSize);
        }
        match self.offset.checked_add(self.size) {
            Some(end) if end <= desc.size => Ok(()),
            _ => Err(GlResourceValidationError::BufferRangeOutOfBounds),
        }
    }
}

#[path = "resource_texture.rs"]
mod resource_texture;
pub(crate) use resource_texture::*;

/// Resource allocation and destruction domain.
pub(crate) trait GlResourceApi: super::GlFamilyApi {
    fn create_buffer_resource(&mut self, desc: GlBufferDesc) -> Result<BufferId, super::GlError>;
    fn create_texture_resource(&mut self, desc: GlTextureDesc)
    -> Result<TextureId, super::GlError>;
    fn create_render_buffer(
        &mut self,
        desc: GlRenderBufferDesc,
    ) -> Result<RenderbufferId, super::GlError>;
    fn destroy_buffer_resource(&mut self, buffer: BufferId) -> Result<(), super::GlError>;
    fn destroy_texture_resource(&mut self, texture: TextureId) -> Result<(), super::GlError>;
    fn destroy_render_buffer(
        &mut self,
        render_buffer: RenderbufferId,
    ) -> Result<(), super::GlError>;
}

#[cfg(test)]
#[path = "resource_tests.rs"]
mod tests;
