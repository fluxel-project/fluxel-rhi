//! Native OpenGL / OpenGL ES context ownership and entry-point evidence.
//!
//! This is deliberately below the platform-neutral [`crate::backend::api`]
//! contract.  WGL and EGL objects never escape this module: a caller hands us a
//! context which it has already made current, and receives a `!Send` executor
//! tied to that thread.  In particular, this is not shared with the WebGL
//! implementation; browser objects have a different lifetime model.
//!
//! The native minimum is desktop GL 4.0 or GLES 3.0.  A newer context is kept
//! at its actual version.  Optional functionality is admitted from *both* the
//! version/extension ledger and the loaded entry-point table, never from a
//! version string alone.

pub(crate) mod buffer_abi;
mod buffer_mapping;
mod clear_texture_direct;
mod clear_texture_upload;
mod context;
mod discovery;
pub(crate) mod draw_abi;
#[cfg(feature = "native-gles-egl")]
pub(crate) mod egl;
mod entry_points;
mod exec_clear_texture;
mod exec_compressed_readback;
mod exec_compressed_upload;
mod exec_compute;
mod exec_copy;
mod exec_framebuffer;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
mod exec_mapping;
mod exec_multi_draw;
mod exec_pbo_readback;
mod exec_pbo_upload;
mod exec_raster;
mod exec_rgb9e5_readback;
mod exec_shader;
mod exec_sync;
mod exec_texture_sub_image_readback;
mod exec_vertex;
pub(crate) mod indexed_blend;
mod indirect_count;
pub(crate) mod indirect_count_emulated;
mod indirect_rebase;
#[cfg(test)]
#[path = "../../tests/backend/native/indirect_repro.rs"]
mod indirect_repro;
pub(crate) mod interop;
mod multi_draw_indirect;
mod multisample_array;
mod multiview;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
mod probe;
mod probes;
mod program_cache;
mod program_resources;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) mod provider;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) mod push_uniform;
mod rasterization;
mod sampler_border;
mod spirv;
mod surface;
mod surface_facts;
pub(crate) mod texture_view;
#[cfg(all(windows, feature = "native-gl-wgl"))]
pub(crate) mod wgl;

pub(crate) use context::{
    CurrentContextGuard, NativeContext, NativeContextKind, NativeContextOwner,
    NativePlatformContext,
};
pub(crate) use entry_points::{NativeEntryPoint, NativeEntryPoints};
pub(crate) use probe::{NativeProfileError, parse_native_profile};

// Real GL-family conformance cases live under `tests` beside the
// other backend fixtures.  The fixture opens its WGL context here, where raw
// window/context ownership is still private, then exercises the public Device
// through the common workload.  It must not turn WGL construction into a
// portable public API merely to make an integration test convenient.
#[cfg(all(test, windows, feature = "native-gl-wgl"))]
#[path = "../../../tests/gl/wgl_core.rs"]
mod wgl_core_conformance;

mod shader_bindings;
mod shader_blocks;
mod shader_compile;
mod shader_interfaces;
mod shader_link;
mod spirv_opaque;
mod spirv_vertex;

mod framebuffer_attachment;

mod resource;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
pub(crate) use resource::native_texture_format_supported;
mod cube_mirror;
mod dummy_binding;
mod stencil_mirror;

mod stencil_extract;

/// Stable native seam tag shared by the immediate lowering and platform
/// currentness compatibility check. Keep this identity separate from ad-hoc
/// caller diagnostics so label changes cannot bypass the driver workaround.
pub(crate) const INDIRECT_COUNT_OPERATION: &str = "multi draw GL indirect count";
