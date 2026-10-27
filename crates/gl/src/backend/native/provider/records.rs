//! Context-owned native object records and execution facts.
//!
//! These types carry validated CPU metadata for the provider but issue no GL
//! calls. Keeping them separate from provider orchestration makes their
//! lifetime boundary explicit: every record dies with its context generation.

use std::collections::BTreeMap;

use crate::backend::api::{
    GlIndexBinding, GlPrimitiveTopology, GlProgramDescriptor, GlRenderTarget, GlVertexLayout,
    ProgramId,
};

/// A linked raster or compute program record with its validated descriptor.
pub(crate) struct NativeProgram {
    pub(crate) generation: u32,
    pub(crate) raw: glow::NativeProgram,
    pub(crate) descriptor: GlProgramDescriptor,
    /// Link-time logical-to-native assignments. Binding flushes consult this
    /// instead of treating an RHI slot as a GL binding point.
    pub(crate) reflection: crate::backend::api::GlProgramReflection,
    /// Link-generation-scoped locations, including optimized-away `None`.
    pub(crate) first_instance_uniform: Option<glow::NativeUniformLocation>,
    pub(crate) uniform_locations: BTreeMap<String, Option<glow::NativeUniformLocation>>,
    /// Required backing sizes for deliberately dummy-bound blocks. `None`
    /// records an absent block, preventing repeated sparse-bind reflection.
    pub(crate) dummy_storage_block_sizes: BTreeMap<u32, Option<u32>>,
    pub(crate) dummy_uniform_block_sizes: BTreeMap<u32, Option<u32>>,
    pub(crate) first_instance_value: Option<u32>,
}

/// One exact indexed GL shader-storage-buffer binding.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NativeStorageBinding {
    pub(crate) buffer: Option<glow::NativeBuffer>,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

/// A created VAO with its structural layout and last recorded index binding.
pub(crate) struct NativeVertexArray {
    pub(crate) generation: u32,
    pub(crate) raw: glow::NativeVertexArray,
    pub(crate) layout: GlVertexLayout,
    pub(crate) index: Option<GlIndexBinding>,
    pub(crate) bindings: Vec<crate::backend::api::GlVertexBufferBinding>,
}

/// A created framebuffer with its validated descriptor.
pub(crate) struct NativeFramebuffer {
    pub(crate) generation: u32,
    pub(crate) raw: glow::NativeFramebuffer,
    pub(crate) descriptor: crate::backend::api::GlFramebufferDescriptor,
    pub(crate) multiview: bool,
}

/// The target a query object last recorded a measurement for.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum QueryTarget {
    SamplesPassed,
    TimeElapsed,
    Timestamp,
    Statistics(u32),
}

pub(crate) struct NativeQuery {
    pub(crate) generation: u32,
    pub(crate) raw: glow::NativeQuery,
    pub(crate) segments: Vec<glow::NativeQuery>,
    pub(crate) invalidated: bool,
    pub(crate) target: Option<QueryTarget>,
    pub(crate) disjoint: bool,
}

/// EXT entry points loaded from the current EGL context.
pub(crate) struct GlesTimerFunctions {
    pub(crate) query_counter: unsafe extern "system" fn(u32, u32),
    pub(crate) query_result: unsafe extern "system" fn(u32, u32, *mut u64),
    pub(crate) bits: u32,
    pub(crate) elapsed_bits: u32,
}

/// Facts of the render pass currently recording on this context.
pub(crate) struct ActivePass {
    pub(crate) target: GlRenderTarget,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) samples: u32,
    /// Exact active attachment numeric classes, retained for active-pass
    /// operations such as `glClearBuffer*` validation.
    pub(crate) color_formats: Vec<Option<crate::backend::api::GlFormat>>,
    pub(crate) depth_stencil_format: Option<crate::backend::api::GlFormat>,
    pub(crate) discard_color: Vec<bool>,
    pub(crate) discard_depth_stencil: Option<(bool, bool)>,
}

/// The raster pipeline installed for the active pass.
pub(crate) struct ActiveRaster {
    /// A compute install or link may supersede it; draws reassert this through
    /// [`super::NativeGlProvider::ensure_program`].
    pub(crate) program: ProgramId,
    /// VAO ownership stays in `NativeGlProvider::bound_vertex_array` because
    /// geometry lowering can independently change that one GL slot.
    pub(crate) topology: GlPrimitiveTopology,
    pub(crate) depth_stencil: Option<crate::backend::api::GlDepthStencilState>,
}
