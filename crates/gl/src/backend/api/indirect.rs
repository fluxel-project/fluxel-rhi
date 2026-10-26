//! Indirect command-buffer record layouts and their validated ranges.
//!
//! The records are the WebGPU-aligned indirect argument layouts this contract
//! accepts, so a caller that uploads WebGPU-shaped arguments needs no repacking.
//! The single, multi, and count paths stay distinct types because they have
//! distinct record shapes, distinct buffer roles, and distinct capability
//! evidence; they are never interchangeable. This module owns the layouts and
//! the range arithmetic only: issuing a command, binding its buffer, and
//! checking the context's capability belong to the provider.
//!
//! A range is legal only when both of its bounds hold, and both live here. The
//! record validators prove a record fits inside the range; the allocation bound
//! proves the range fits inside the buffer the range names. Checking only the
//! first lets a caller describe a window wider than its allocation, which makes
//! the driver read an argument list that was never allocated -- a silent
//! out-of-bounds read rather than a rejection. The two checks are therefore
//! separate functions that every indirect entry point applies at the same point
//! in the same order, so the native provider, the browser provider, and the
//! recorder cannot diverge on either the rule or its error.

use super::{GlBufferDesc, GlBufferRange, GlError, GlFamilyApi};

/// ABI of the non-indexed draw record: four u32 words.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlDrawArraysIndirectAbi {
    pub count: u32,
    pub instance_count: u32,
    pub first: u32,
    pub base_instance: u32,
}
/// ABI of the indexed draw record: four u32 words plus a signed base vertex.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlDrawElementsIndirectAbi {
    pub count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub base_instance: u32,
}
/// ABI of the dispatch record: three u32 work-group counts.
///
/// This is exactly the WebGPU dispatch-indirect argument layout. A wider
/// platform record that carries an extra word is deliberately unreachable: one
/// command has one record shape here, so a stale wider record cannot silently
/// have its fourth word reinterpreted as a work-group count.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlDispatchIndirectAbi {
    pub workgroup_count: [u32; 3],
}

impl GlDispatchIndirectAbi {
    /// Exact record size in bytes, taken from the layout instead of restated.
    pub(crate) const SIZE: u64 = std::mem::size_of::<Self>() as u64;
}

fn validate_indirect_range(range: GlBufferRange, operation: &'static str) -> Result<(), GlError> {
    if range.size == 0 || !range.offset.is_multiple_of(4) || !range.size.is_multiple_of(4) {
        Err(GlError::Validation {
            operation,
            message: "indirect buffer ranges must be nonempty and 4-byte aligned".into(),
        })
    } else {
        Ok(())
    }
}

/// Rejects an indirect range that is not wholly inside the buffer it names.
///
/// The second half of a range's contract: `validate_indirect_range` proves the
/// range is a usable window, and this proves the window is inside its
/// allocation. A range past the end of the buffer makes the driver read records
/// beyond the allocation, which no driver reports and no caller can observe, so
/// the bound has to be rejected before the buffer is bound. Every indirect
/// entry point calls this immediately after resolving the buffer descriptor,
/// which is the earliest point the allocation's size is known and the last
/// point before any binding changes.
///
/// Both providers and the recorder call this one function rather than restating
/// the bound, so they cannot disagree about which ranges are legal or about the
/// error a rejected one produces.
pub(crate) fn validate_indirect_allocation(
    operation: &'static str,
    range: GlBufferRange,
    desc: GlBufferDesc,
) -> Result<(), GlError> {
    range.validate_for(desc).map_err(|_| GlError::Validation {
        operation,
        message: "indirect buffer range is outside the allocation".into(),
    })
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GlIndirectAbi {
    NonIndexed,
    Indexed,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlIndirectCommandRange {
    pub range: GlBufferRange,
    pub command_offset: u64,
    pub draw_count: u32,
    pub stride: u32,
    pub abi: GlIndirectAbi,
}
impl GlIndirectCommandRange {
    pub(crate) fn validate(self, operation: &'static str) -> Result<(), GlError> {
        validate_indirect_range(self.range, operation)?;
        let record = match self.abi {
            GlIndirectAbi::NonIndexed => 16,
            GlIndirectAbi::Indexed => 20,
        };
        if self.draw_count == 0
            || !self.command_offset.is_multiple_of(4)
            || self.stride != 0 && (self.stride < record || !self.stride.is_multiple_of(4))
        {
            return Err(GlError::Validation {
                operation,
                message: "indirect offset, count, or stride violates the selected ABI".into(),
            });
        }
        let stride = if self.stride == 0 {
            record
        } else {
            self.stride
        } as u64;
        let needed = self
            .command_offset
            .checked_add((u64::from(self.draw_count) - 1).saturating_mul(stride))
            .and_then(|v| v.checked_add(u64::from(record)));
        if needed.is_none_or(|v| v > self.range.size) {
            return Err(GlError::Validation {
                operation,
                message: "indirect commands exceed their buffer range".into(),
            });
        }
        Ok(())
    }
}
/// One dispatch record selected inside a command buffer range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlDispatchIndirectCommand {
    pub range: GlBufferRange,
    pub command_offset: u64,
}
impl GlDispatchIndirectCommand {
    /// Rejects a record that is misaligned or not wholly inside its range.
    ///
    /// The one-record path has no stride and no count: a dispatch record is
    /// always read whole, so the only failure modes are alignment and extent.
    pub(crate) fn validate(self, operation: &'static str) -> Result<(), GlError> {
        validate_indirect_range(self.range, operation)?;
        if !self.command_offset.is_multiple_of(4)
            || self
                .command_offset
                .checked_add(GlDispatchIndirectAbi::SIZE)
                .is_none_or(|end| end > self.range.size)
        {
            return Err(GlError::Validation {
                operation,
                message: "indirect dispatch record leaves its buffer range".into(),
            });
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GlIndirectCountRange {
    pub range: GlBufferRange,
    pub count_offset: u64,
    pub max_draw_count: u32,
}
impl GlIndirectCountRange {
    /// Rejects a count word that is misaligned, absent, or not wholly in range.
    ///
    /// The operation name is the *caller's*, on [`GlDispatchIndirectCommand::validate`]'s
    /// terms and for the same reason: an error that names a verb the frame never
    /// issued is a worse diagnostic than no name at all, and the caller is the
    /// only layer that still knows which verb it was.  This one used to hardcode
    /// its own name, and the cost was two spellings of the same verb reaching an
    /// operator -- `multi_draw_indirect_count` from here and
    /// `multi-draw-indirect-count` from the recorder that called it -- because a
    /// hardcoded name is a second place the spelling is written down.
    pub(crate) fn validate(self, operation: &'static str) -> Result<(), GlError> {
        validate_indirect_range(self.range, operation)?;
        if self.max_draw_count == 0
            || !self.count_offset.is_multiple_of(4)
            || self
                .count_offset
                .checked_add(4)
                .is_none_or(|v| v > self.range.size)
        {
            return Err(GlError::Validation {
                operation,
                message: "count range is not a 4-byte in-range value".into(),
            });
        }
        Ok(())
    }
}

pub(crate) trait GlDrawIndirectApi: GlFamilyApi {
    fn draw_indirect(&mut self, command: GlIndirectCommandRange) -> Result<(), GlError>;
}
pub(crate) trait GlDispatchIndirectApi: GlFamilyApi {
    /// Issues one work-group triple read from an indirect command buffer.
    fn dispatch_indirect(&mut self, command: GlDispatchIndirectCommand) -> Result<(), GlError>;
}
pub(crate) trait GlMultiDrawIndirectApi: GlFamilyApi {
    fn multi_draw_indirect(&mut self, commands: GlIndirectCommandRange) -> Result<(), GlError>;
}
pub(crate) trait GlMultiDrawCountApi: GlFamilyApi {
    fn multi_draw_indirect_count(
        &mut self,
        commands: GlIndirectCommandRange,
        count: GlIndirectCountRange,
    ) -> Result<(), GlError>;
}

#[cfg(test)]
#[path = "../../tests/backend/api/indirect__tests.rs"]
mod tests;
