//! Canonical immutable raster-pipeline state for one GL context.
//!

//! These structural IDs are allocated only by the owning state mirror. They
//! describe already-created private GL objects and fixed-function leaves; this
//! module neither binds GL nor owns command ordering.

use std::collections::HashMap;

use crate::backend::gl::api::{
    ContextStamp, GlColorTargetState, GlCullMode, GlDepthStencilState, GlFrontFace,
    GlMultisampleState, GlObjectKind, GlPrimitiveTopology, GlRasterPipeline, GlScissorRect,
    GlViewport, ObjectIdentity, ProgramId,
};

use super::GlObjectKey;

/// A canonical immutable block prepared during phase A.  It is assigned by
/// the private pipeline/binding table, never synthesized from a lossy hash.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) enum CanonicalBlockId {
    /// Private monotonically allocated immutable-state block.  Such blocks are
    /// owned by this one `ContextState`, which is discarded on context loss.
    Allocated(u64),
    /// A real GL object identity.  Do not reduce this to a native name or an
    /// allocation slot: browser registries deliberately reuse slots and a
    /// restored context may reuse the same native names.
    Object {
        context: ContextStamp,
        slot: u32,
        generation: u32,
    },
    /// `glBindBufferRange` state includes the selected range, not merely the
    /// buffer.  Keeping all fields here prevents an equal-buffer fast path
    /// from skipping a legitimate offset/size update.
    UniformRange {
        context: ContextStamp,
        slot: u32,
        generation: u32,
        offset: u32,
        size: u32,
    },
}
impl CanonicalBlockId {
    pub(crate) const fn new(value: u64) -> Self {
        Self::Allocated(value)
    }
    pub(crate) const fn object<K: GlObjectKind>(identity: ObjectIdentity<K>) -> Self {
        Self::Object {
            context: identity.context,
            slot: identity.slot,
            generation: identity.generation,
        }
    }
    pub(crate) const fn uniform_range<K: GlObjectKind>(
        buffer: ObjectIdentity<K>,
        offset: u32,
        size: u32,
    ) -> Self {
        Self::UniformRange {
            context: buffer.context,
            slot: buffer.slot,
            generation: buffer.generation,
            offset,
            size,
        }
    }
}

/// Canonical blocks that a pipeline installs independently in GL.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RasterPipelineBlocks {
    pub(crate) program: CanonicalBlockId,
    pub(crate) raster: CanonicalBlockId,
    pub(crate) depth_stencil: CanonicalBlockId,
    pub(crate) blend: CanonicalBlockId,
    pub(crate) multisample: CanonicalBlockId,
}

/// Per-context structural canonicalization for immutable raster pipeline
/// leaves.  These IDs are deliberately *not* creation serials: a pipeline
/// switch can only skip an unchanged GL leaf when equal descriptors intern to
/// the same value.  `HashMap` equality resolves collisions, so no lossy hash
/// ever becomes a state-cache identity.
#[derive(Default)]
pub(crate) struct RasterPipelineBlockInterner {
    next: u64,
    programs: HashMap<ProgramId, CanonicalBlockId>,
    rasters: HashMap<RasterBlock, CanonicalBlockId>,
    depth_stencils: HashMap<Option<GlDepthStencilState>, CanonicalBlockId>,
    blends: HashMap<BlendBlock, CanonicalBlockId>,
    multisamples: HashMap<GlMultisampleState, CanonicalBlockId>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct RasterBlock {
    rasterization: super::super::api::GlRasterizationState,
    topology: GlPrimitiveTopology,
    cull_mode: GlCullMode,
    front_face: GlFrontFace,
    viewport: GlViewport,
    scissor: Option<GlScissorRect>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct BlendBlock {
    color_targets: Vec<GlColorTargetState>,
    blend_constant: [u32; 4],
}

impl RasterPipelineBlockInterner {
    pub(crate) fn new() -> Self {
        Self {
            // Zero stays invalid so a malformed transport/default value cannot
            // accidentally compare equal to a real canonical block.
            next: 1,
            ..Self::default()
        }
    }

    pub(crate) fn intern(
        &mut self,
        pipeline: &GlRasterPipeline,
        operation: &'static str,
    ) -> Result<RasterPipelineBlocks, &'static str> {
        let state = &pipeline.state;
        Ok(RasterPipelineBlocks {
            program: self.intern_program(pipeline.program, operation)?,
            raster: self.intern_raster(
                RasterBlock {
                    rasterization: state.rasterization,
                    topology: state.topology,
                    cull_mode: state.cull_mode,
                    front_face: state.front_face,
                    viewport: state.viewport,
                    scissor: state.scissor,
                },
                operation,
            )?,
            depth_stencil: self.intern_depth_stencil(state.depth_stencil, operation)?,
            blend: self.intern_blend(
                BlendBlock {
                    color_targets: state.color_targets.clone(),
                    blend_constant: state.blend_constant,
                },
                operation,
            )?,
            multisample: self.intern_multisample(state.multisample, operation)?,
        })
    }

    fn intern_program(
        &mut self,
        value: ProgramId,
        operation: &'static str,
    ) -> Result<CanonicalBlockId, &'static str> {
        intern(&mut self.next, &mut self.programs, value, operation)
    }
    fn intern_raster(
        &mut self,
        value: RasterBlock,
        operation: &'static str,
    ) -> Result<CanonicalBlockId, &'static str> {
        intern(&mut self.next, &mut self.rasters, value, operation)
    }
    fn intern_depth_stencil(
        &mut self,
        value: Option<GlDepthStencilState>,
        operation: &'static str,
    ) -> Result<CanonicalBlockId, &'static str> {
        intern(&mut self.next, &mut self.depth_stencils, value, operation)
    }
    fn intern_blend(
        &mut self,
        value: BlendBlock,
        operation: &'static str,
    ) -> Result<CanonicalBlockId, &'static str> {
        intern(&mut self.next, &mut self.blends, value, operation)
    }
    fn intern_multisample(
        &mut self,
        value: GlMultisampleState,
        operation: &'static str,
    ) -> Result<CanonicalBlockId, &'static str> {
        intern(&mut self.next, &mut self.multisamples, value, operation)
    }
}

fn intern<T: Eq + std::hash::Hash>(
    next: &mut u64,
    table: &mut HashMap<T, CanonicalBlockId>,
    value: T,
    operation: &'static str,
) -> Result<CanonicalBlockId, &'static str> {
    if let Some(id) = table.get(&value) {
        return Ok(*id);
    }
    let id = CanonicalBlockId::new(*next);
    *next = next.checked_add(1).ok_or(operation)?;
    table.insert(value, id);
    Ok(id)
}

/// Phase-A's immutable pipeline packet. `identity` is the backend allocation
/// identity, so the exact-hit fast path cannot confuse a reused native name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RasterPipelinePacket {
    pub(crate) identity: GlObjectKey,
    pub(crate) blocks: RasterPipelineBlocks,
}
