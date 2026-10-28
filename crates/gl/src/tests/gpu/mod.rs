//! GL-family hardware regressions organized by behavior and resource contract.
//!
//! Private context seams remain unit tests; each platform keeps its own entry
//! and shader/readback assertions. Shared setup lives in support.
use crate::{Gl, GlInstance};
use fluxel_rhi_core::api::*;
use fluxel_rhi_core::backend::*;
use std::sync::Arc;

mod support;
#[cfg(not(target_arch = "wasm32"))]
use support::block_on;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
use support::webgl2_device;
use support::{open, single_sample_state, triangle_list_state, unblended_target};
#[path = "raster/attachment_subresource.rs"]
mod attachment_subresource;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "bindings/fixed/1d.rs"]
mod binding_1d;
#[path = "bindings/fixed/3d.rs"]
mod binding_3d;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/fixed/barrier.rs"]
mod binding_barrier;
#[path = "bindings/fixed/cube.rs"]
mod binding_cube;
#[path = "bindings/fixed/cube_array.rs"]
mod binding_cube_array;
#[cfg(any(
    all(target_arch = "wasm32", feature = "webgl2"),
    all(feature = "native-gles-egl", not(target_arch = "wasm32"))
))]
#[path = "bindings/fixed/cube_mirror.rs"]
mod binding_cube_mirror;
#[path = "bindings/fixed/cube_storage.rs"]
mod binding_cube_storage;
#[path = "bindings/depth_stencil/depth_load.rs"]
mod binding_depth_load;
#[path = "bindings/direct/buffer_array.rs"]
mod binding_direct_buffer_array;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "bindings/direct/image_array.rs"]
mod binding_direct_image_array;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "bindings/direct/image_units.rs"]
mod binding_direct_image_units;
#[path = "bindings/direct/sampler_array.rs"]
mod binding_direct_sampler_array;
#[path = "bindings/indexed/bias.rs"]
mod binding_dynamic_bias;
#[path = "bindings/indexed/cube_shadow.rs"]
mod binding_dynamic_cube_shadow;
#[path = "bindings/indexed/diagonal.rs"]
mod binding_dynamic_diagonal;
#[path = "bindings/indexed/gather.rs"]
mod binding_dynamic_gather;
#[path = "bindings/indexed/helper.rs"]
mod binding_dynamic_helper;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/indexed/image.rs"]
mod binding_dynamic_image;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/indexed/image_load.rs"]
mod binding_dynamic_image_load;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/indexed/image_mixed.rs"]
mod binding_dynamic_image_mixed;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "bindings/indexed/image_query.rs"]
mod binding_dynamic_image_query;
#[path = "bindings/indexed/implicit.rs"]
mod binding_dynamic_implicit;
#[path = "bindings/indexed/pairs.rs"]
mod binding_dynamic_pairs;
#[path = "bindings/indexed/query.rs"]
mod binding_dynamic_query;
#[path = "bindings/indexed/query_layers.rs"]
mod binding_dynamic_query_layers;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "bindings/indexed/query_samples.rs"]
mod binding_dynamic_query_samples;
#[path = "bindings/indexed/sampled.rs"]
mod binding_dynamic_sampled;
#[path = "bindings/indexed/sampled_array.rs"]
mod binding_dynamic_sampled_array;
#[path = "bindings/indexed/sampled_depth.rs"]
mod binding_dynamic_sampled_depth;
#[path = "bindings/indexed/sampled_float.rs"]
mod binding_dynamic_sampled_float;
#[path = "bindings/indexed/sampled_volume.rs"]
mod binding_dynamic_sampled_volume;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/indexed/storage.rs"]
mod binding_dynamic_storage;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/indexed/storage_compare.rs"]
mod binding_dynamic_storage_compare;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/indexed/storage_signed.rs"]
mod binding_dynamic_storage_signed;
#[path = "bindings/indexed/ubo.rs"]
mod binding_dynamic_ubo;
#[path = "bindings/fixed/helper_pairs.rs"]
mod binding_helper_pairs;
#[path = "bindings/fixed/integer_compare.rs"]
mod binding_integer_compare;
#[path = "bindings/fixed/pairs.rs"]
mod binding_pairs;
#[path = "bindings/partial/array.rs"]
mod binding_partial_array;
#[path = "bindings/partial/compare.rs"]
mod binding_partial_compare;
#[path = "bindings/partial/cube.rs"]
mod binding_partial_cube;
#[path = "bindings/partial/depth.rs"]
mod binding_partial_depth;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/partial/image.rs"]
mod binding_partial_image;
#[path = "bindings/partial/integer.rs"]
mod binding_partial_integer;
#[path = "bindings/partial/pairs.rs"]
mod binding_partial_pairs;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/partial/ssbo.rs"]
mod binding_partial_ssbo;
#[path = "bindings/partial/ubo.rs"]
mod binding_partial_ubo;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/runtime/divergent_storage.rs"]
mod binding_runtime_divergent_storage;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/runtime/hole.rs"]
mod binding_runtime_hole;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/runtime/image.rs"]
mod binding_runtime_image;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/runtime/prefix.rs"]
mod binding_runtime_prefix;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/runtime/uniform.rs"]
mod binding_runtime_uniform;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "bindings/runtime/webgl.rs"]
mod binding_runtime_webgl;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "bindings/runtime/webgl_uniform.rs"]
mod binding_runtime_webgl_uniform;
#[path = "bindings/fixed/sampling_ops.rs"]
mod binding_sampling_ops;
#[path = "bindings/validation/shader_validation.rs"]
mod binding_shader_validation;
#[path = "bindings/fixed/shared_sampler.rs"]
mod binding_shared_sampler;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/fixed/ssbo.rs"]
mod binding_ssbo;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/fixed/stage_images.rs"]
mod binding_stage_images;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/fixed/static_image.rs"]
mod binding_static_image;
#[path = "bindings/fixed/static_pairs.rs"]
mod binding_static_pairs;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "bindings/fixed/static_storage.rs"]
mod binding_static_storage;
#[path = "bindings/depth_stencil/stencil_array.rs"]
mod binding_stencil_array;
#[path = "bindings/depth_stencil/stencil_load.rs"]
mod binding_stencil_load;
#[path = "bindings/fixed/unit_switch.rs"]
mod binding_unit_switch;
#[path = "bindings/validation/validation.rs"]
mod binding_validation;
#[path = "textures/transfer/clear_nonrenderable.rs"]
mod clear_nonrenderable;
#[path = "support/immediate.rs"]
mod common;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "textures/compressed/3d.rs"]
mod compressed_3d;
#[path = "textures/compressed/etc2.rs"]
mod compressed_etc2;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/compressed/gather.rs"]
mod compressed_gather;
#[path = "textures/compressed/mips.rs"]
mod compressed_mips;
#[path = "textures/compressed/pbo.rs"]
mod compressed_pbo;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/compressed/readback.rs"]
mod compressed_readback;
#[path = "textures/compressed/readback_matrix.rs"]
mod compressed_readback_matrix;
#[path = "textures/compressed/rebind.rs"]
mod compressed_rebind;
#[path = "textures/compressed/astc.rs"]
mod core_compressed_astc;
#[path = "textures/compressed/bptc.rs"]
mod core_compressed_bptc;
#[path = "textures/compressed/etc_matrix.rs"]
mod core_compressed_etc_matrix;
#[path = "textures/compressed/rgtc.rs"]
mod core_compressed_rgtc;
#[path = "raster/polygon_offset_clamp.rs"]
mod core_polygon_offset_clamp;
#[path = "raster/zero_stride.rs"]
mod core_zero_stride;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "textures/cube/edges.rs"]
mod cube_edges;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/cube/faces.rs"]
mod cube_faces;
#[path = "textures/cube/mirror_mips.rs"]
mod cube_mirror_mips;
#[path = "textures/depth/depth16.rs"]
mod depth16;
#[path = "textures/depth/depth_blit.rs"]
mod depth_blit;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "raster/desktop_multiview.rs"]
mod desktop_multiview;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/msaa/array_resolve_sampling.rs"]
mod diagnostic_msaa_array;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "textures/readback/direct_color_readback.rs"]
mod direct_color_readback;
#[path = "shaders/artifacts/dual_source_blend.rs"]
mod direct_dual_source_blend;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "shaders/artifacts/glsl_f64_vertex.rs"]
mod direct_glsl_f64_vertex;
#[path = "shaders/artifacts/offsets.rs"]
mod direct_offsets;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "shaders/artifacts/shader_artifacts.rs"]
mod direct_shader_artifacts;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "shaders/artifacts/spirv_depth.rs"]
mod direct_spirv_depth;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "shaders/artifacts/spirv_pairs.rs"]
mod direct_spirv_pairs;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "shaders/artifacts/spirv_storage.rs"]
mod direct_spirv_storage;
#[path = "raster/dual_source_blend.rs"]
mod dual_source_blend;
#[path = "raster/dynamic_state.rs"]
mod dynamic;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/msaa/indexed_array_sampling.rs"]
mod dynamic_msaa_array;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/msaa/format_support.rs"]
mod format_msaa;
#[path = "raster/fragment_origin.rs"]
mod fragment_origin;
#[cfg(feature = "native-gles-egl")]
#[path = "textures/formats/gles_exact_alias.rs"]
mod gles_exact_alias;
#[cfg(feature = "native-gles-egl")]
#[path = "textures/formats/gles_float_alias.rs"]
mod gles_float_alias;
#[cfg(feature = "native-gles-egl")]
#[path = "queries/gles_occlusion.rs"]
mod gles_occlusion;
#[cfg(feature = "native-gles-egl")]
#[path = "textures/transfer/gles_packed_transfer.rs"]
mod gles_packed_transfer;
#[cfg(feature = "native-gles-egl")]
#[path = "textures/readback/gles_plane_readback.rs"]
mod gles_plane_readback;
#[cfg(feature = "native-gles-egl")]
#[path = "textures/depth/gles_standalone_depth.rs"]
mod gles_standalone_depth;
#[cfg(feature = "native-gles-egl")]
#[path = "queries/gles_timestamp.rs"]
mod gles_timestamp;
#[path = "raster/independent_resolve.rs"]
mod independent_resolve;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "raster/indirect.rs"]
mod indirect;
#[path = "textures/transfer/layer_blit.rs"]
mod layer_blit;
#[path = "execution/mapping.rs"]
mod mapping;
#[path = "raster/mrt_hole.rs"]
mod mrt_hole;
use mrt_hole::mrt_hole;
#[path = "raster/mrt_indexed.rs"]
mod mrt_indexed;
#[path = "raster/mrt_resolve.rs"]
mod mrt_resolve;
#[path = "surfaces/mrt_surface_resolve.rs"]
mod mrt_surface_resolve;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "textures/msaa/array_formats.rs"]
mod msaa_array_formats;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/msaa/array_sampling.rs"]
mod msaa_array_sampling;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/msaa/copy.rs"]
mod msaa_copy;
#[path = "raster/multiview.rs"]
mod multiview;
#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    feature = "native-gles-egl",
    all(target_arch = "wasm32", feature = "webgl2")
))]
#[path = "queries/multiview_timestamp.rs"]
mod multiview_timestamp;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "textures/readback/native_color_readback.rs"]
mod native_color_readback;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/transfer/native_integer_copy.rs"]
mod native_integer_copy;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "execution/native_interop.rs"]
mod native_interop;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "textures/formats/native_mixed_srgb.rs"]
mod native_mixed_srgb;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/formats/native_srgb.rs"]
mod native_srgb;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "surfaces/native_surface_features.rs"]
mod native_surface_features;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/formats/native_view.rs"]
mod native_view;
#[path = "raster/ordinary_clear.rs"]
mod ordinary_clear;
#[path = "textures/formats/ordinary_formats.rs"]
mod ordinary_formats;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "textures/transfer/packed_plane_copy.rs"]
mod packed_plane_copy;
#[path = "textures/readback/pbo_readback.rs"]
mod pbo_readback;
#[path = "textures/readback/pbo_upload.rs"]
mod pbo_upload;
#[path = "raster/pipeline_override.rs"]
mod pipeline_override;
#[path = "queries/pipeline_statistics.rs"]
mod pipeline_statistics;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "raster/program_cache.rs"]
mod program_cache;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "raster/push_compute.rs"]
mod push_compute;
#[path = "raster/push_constants.rs"]
mod push_constants;
#[path = "raster/push_matrix.rs"]
mod push_matrix;
#[cfg(feature = "native-gles-egl")]
#[path = "queries/query_stencil_extract.rs"]
mod query_stencil_extract;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "raster/raster_modes.rs"]
mod raster_modes;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/readback/rgb9e5_readback.rs"]
mod rgb9e5_readback;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "raster/sample_mask.rs"]
mod sample_mask;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "raster/sampler_border.rs"]
mod sampler_border;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "raster/sampler_border_integer.rs"]
mod sampler_border_integer;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/readback/snorm_endpoint_readback.rs"]
mod snorm_endpoint_readback;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "surfaces/srgb_surface.rs"]
mod srgb_surface;
#[path = "textures/stencil/stencil_blit.rs"]
mod stencil_blit;
#[cfg(feature = "native-gles-egl")]
#[path = "textures/stencil/stencil_extract_mask.rs"]
mod stencil_extract_mask;
#[cfg(feature = "native-gles-egl")]
#[path = "textures/stencil/stencil_extract_mrt.rs"]
mod stencil_extract_mrt;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/stencil/stencil_transfer.rs"]
mod stencil_transfer;
#[path = "execution/step_checkpoints.rs"]
mod step_checkpoints;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/storage/3d.rs"]
mod storage_3d;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/storage/array.rs"]
mod storage_array;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "textures/storage/formats.rs"]
mod storage_formats;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "surfaces/surface_features.rs"]
mod surface_features;
#[path = "queries/timestamp.rs"]
mod timestamp;
#[path = "execution/transition_validation.rs"]
mod transition_validation;
#[path = "textures/readback/unorm_readback.rs"]
mod unorm_readback;
#[path = "raster/vertex_formats.rs"]
mod vertex_formats;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "platform/webgl/clear_masks.rs"]
mod webgl2_clear_masks;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "platform/webgl/raster_features.rs"]
mod webgl2_raster_features;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "diagnostics/webgl_depth16_all_codes.rs"]
mod webgl_d16_exhaustive;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "textures/depth/webgl_depth16.rs"]
mod webgl_depth16;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "textures/depth/webgl_depth16_array.rs"]
mod webgl_depth16_array;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "textures/depth/webgl_depth32.rs"]
mod webgl_depth32;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "textures/transfer/webgl_integer_copy.rs"]
mod webgl_integer_copy;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "platform/webgl/multisampled_sampled_reject.rs"]
mod webgl_multisampled_sampled_reject;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "queries/webgl_query.rs"]
mod webgl_query;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "platform/webgl/raster_modes.rs"]
mod webgl_raster_modes;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "textures/stencil/webgl_stencil_array.rs"]
mod webgl_stencil_array;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "textures/depth/webgl_stencil_depth_preserve.rs"]
mod webgl_stencil_depth_preserve;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "textures/stencil/webgl_stencil_transfer.rs"]
mod webgl_stencil_transfer;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "textures/readback/webgl_unorm_feedback.rs"]
mod webgl_unorm_feedback;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "platform/webgl/zero_stride.rs"]
mod webgl_zero_stride;

#[path = "execution/buffer_clear_readback.rs"]
mod buffer_clear_readback;
