// Fixture registration fragment, included in core_raster_features to retain
// its existing test namespace and shared parent harness.

#[path = "core_independent_resolve.rs"]
mod independent_resolve;
#[path = "core_step_checkpoints.rs"]
mod step_checkpoints;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_direct_glsl_f64_vertex.rs"]
mod direct_glsl_f64_vertex;

#[path = "core_attachment_subresource.rs"]
mod attachment_subresource;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_binding_1d.rs"]
mod binding_1d;
#[path = "core_binding_3d.rs"]
mod binding_3d;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_barrier.rs"]
mod binding_barrier;
#[path = "core_binding_cube.rs"]
mod binding_cube;
#[path = "core_binding_cube_array.rs"]
mod binding_cube_array;
#[cfg(any(
    all(target_arch = "wasm32", feature = "webgl2"),
    all(feature = "native-gles-egl", not(target_arch = "wasm32"))
))]
#[path = "core_binding_cube_mirror.rs"]
mod binding_cube_mirror;
#[path = "core_binding_cube_storage.rs"]
mod binding_cube_storage;
#[path = "core_binding_depth_load.rs"]
mod binding_depth_load;
#[path = "core_binding_direct_buffer_array.rs"]
mod binding_direct_buffer_array;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "core_binding_direct_image_array.rs"]
mod binding_direct_image_array;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "core_binding_direct_image_units.rs"]
mod binding_direct_image_units;
#[path = "core_binding_direct_sampler_array.rs"]
mod binding_direct_sampler_array;
#[path = "core_binding_dynamic_bias.rs"]
mod binding_dynamic_bias;
#[path = "core_binding_dynamic_cube_shadow.rs"]
mod binding_dynamic_cube_shadow;
#[path = "core_binding_dynamic_diagonal.rs"]
mod binding_dynamic_diagonal;
#[path = "core_binding_dynamic_gather.rs"]
mod binding_dynamic_gather;
#[path = "core_binding_dynamic_helper.rs"]
mod binding_dynamic_helper;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_image.rs"]
mod binding_dynamic_image;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_image_load.rs"]
mod binding_dynamic_image_load;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_image_mixed.rs"]
mod binding_dynamic_image_mixed;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_image_query.rs"]
mod binding_dynamic_image_query;
#[path = "core_binding_dynamic_implicit.rs"]
mod binding_dynamic_implicit;
#[path = "core_binding_dynamic_pairs.rs"]
mod binding_dynamic_pairs;
#[path = "core_binding_dynamic_query.rs"]
mod binding_dynamic_query;
#[path = "core_binding_dynamic_query_layers.rs"]
mod binding_dynamic_query_layers;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_query_samples.rs"]
mod binding_dynamic_query_samples;
#[path = "core_binding_dynamic_sampled.rs"]
mod binding_dynamic_sampled;
#[path = "core_binding_dynamic_sampled_array.rs"]
mod binding_dynamic_sampled_array;
#[path = "core_binding_dynamic_sampled_depth.rs"]
mod binding_dynamic_sampled_depth;
#[path = "core_binding_dynamic_sampled_float.rs"]
mod binding_dynamic_sampled_float;
#[path = "core_binding_dynamic_sampled_volume.rs"]
mod binding_dynamic_sampled_volume;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_storage.rs"]
mod binding_dynamic_storage;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_storage_compare.rs"]
mod binding_dynamic_storage_compare;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_dynamic_storage_signed.rs"]
mod binding_dynamic_storage_signed;
#[path = "core_binding_dynamic_ubo.rs"]
mod binding_dynamic_ubo;
#[path = "core_binding_helper_pairs.rs"]
mod binding_helper_pairs;
#[path = "core_binding_integer_compare.rs"]
mod binding_integer_compare;
#[path = "core_binding_pairs.rs"]
mod binding_pairs;
#[path = "core_binding_partial_array.rs"]
mod binding_partial_array;
#[path = "core_binding_partial_compare.rs"]
mod binding_partial_compare;
#[path = "core_binding_partial_cube.rs"]
mod binding_partial_cube;
#[path = "core_binding_partial_depth.rs"]
mod binding_partial_depth;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_partial_image.rs"]
mod binding_partial_image;
#[path = "core_binding_partial_integer.rs"]
mod binding_partial_integer;
#[path = "core_binding_partial_pairs.rs"]
mod binding_partial_pairs;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_partial_ssbo.rs"]
mod binding_partial_ssbo;
#[path = "core_binding_partial_ubo.rs"]
mod binding_partial_ubo;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_runtime_divergent_storage.rs"]
mod binding_runtime_divergent_storage;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_runtime_hole.rs"]
mod binding_runtime_hole;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_runtime_image.rs"]
mod binding_runtime_image;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_runtime_prefix.rs"]
mod binding_runtime_prefix;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_runtime_uniform.rs"]
mod binding_runtime_uniform;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_binding_runtime_webgl.rs"]
mod binding_runtime_webgl;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_binding_runtime_webgl_uniform.rs"]
mod binding_runtime_webgl_uniform;
#[path = "core_binding_sampling_ops.rs"]
mod binding_sampling_ops;
#[path = "core_binding_shader_validation.rs"]
mod binding_shader_validation;
#[path = "core_binding_shared_sampler.rs"]
mod binding_shared_sampler;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_ssbo.rs"]
mod binding_ssbo;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_stage_images.rs"]
mod binding_stage_images;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_static_image.rs"]
mod binding_static_image;
#[path = "core_binding_static_pairs.rs"]
mod binding_static_pairs;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_binding_static_storage.rs"]
mod binding_static_storage;
#[path = "core_binding_stencil_array.rs"]
mod binding_stencil_array;
#[path = "core_binding_stencil_load.rs"]
mod binding_stencil_load;
#[path = "core_binding_unit_switch.rs"]
mod binding_unit_switch;
#[path = "core_binding_validation.rs"]
mod binding_validation;
#[path = "core_clear_nonrenderable.rs"]
mod clear_nonrenderable;
#[path = "../common/mod.rs"]
mod common;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_compressed_3d.rs"]
mod compressed_3d;
#[path = "core_compressed_bc1_pbo.rs"]
mod compressed_bc1_pbo;
#[path = "core_compressed_etc2.rs"]
mod compressed_etc2;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_compressed_gather.rs"]
mod compressed_gather;
#[path = "core_compressed_mips.rs"]
mod compressed_mips;
#[path = "core_compressed_pbo.rs"]
mod compressed_pbo;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_compressed_readback.rs"]
mod compressed_readback;
#[path = "core_compressed_readback_matrix.rs"]
mod compressed_readback_matrix;
#[path = "core_compressed_rebind.rs"]
mod compressed_rebind;
#[path = "core_compressed_astc.rs"]
mod core_compressed_astc;
#[path = "core_compressed_bptc.rs"]
mod core_compressed_bptc;
#[path = "core_compressed_etc_matrix.rs"]
mod core_compressed_etc_matrix;
#[path = "core_compressed_rgtc.rs"]
mod core_compressed_rgtc;
#[path = "core_polygon_offset_clamp.rs"]
mod core_polygon_offset_clamp;
#[path = "core_zero_stride.rs"]
mod core_zero_stride;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_cube_edges.rs"]
mod cube_edges;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_cube_faces.rs"]
mod cube_faces;
#[path = "core_cube_mirror_mips.rs"]
mod cube_mirror_mips;
#[path = "core_depth16.rs"]
mod depth16;
#[path = "core_depth_blit.rs"]
mod depth_blit;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_desktop_multiview.rs"]
mod desktop_multiview;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_diagnostic_msaa_array.rs"]
mod diagnostic_msaa_array;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_direct_color_readback.rs"]
mod direct_color_readback;
#[path = "core_direct_dual_source_blend.rs"]
mod direct_dual_source_blend;
#[path = "core_direct_offsets.rs"]
mod direct_offsets;
#[path = "core_dual_source_blend.rs"]
mod dual_source_blend;
#[path = "core_dynamic_features.rs"]
mod dynamic;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_dynamic_msaa_array.rs"]
mod dynamic_msaa_array;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_format_msaa.rs"]
mod format_msaa;
#[path = "core_fragment_origin.rs"]
mod fragment_origin;
#[cfg(feature = "native-gles-egl")]
#[path = "core_gles_exact_alias.rs"]
mod gles_exact_alias;
#[cfg(feature = "native-gles-egl")]
#[path = "core_gles_float_alias.rs"]
mod gles_float_alias;
#[cfg(feature = "native-gles-egl")]
#[path = "core_gles_occlusion.rs"]
mod gles_occlusion;
#[cfg(feature = "native-gles-egl")]
#[path = "core_gles_packed_transfer.rs"]
mod gles_packed_transfer;
#[cfg(feature = "native-gles-egl")]
#[path = "core_gles_plane_readback.rs"]
mod gles_plane_readback;
#[cfg(feature = "native-gles-egl")]
#[path = "core_gles_standalone_depth.rs"]
mod gles_standalone_depth;
#[cfg(feature = "native-gles-egl")]
#[path = "core_gles_timestamp.rs"]
mod gles_timestamp;
#[path = "core_gles_unorm_readback.rs"]
mod gles_unorm_readback;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_indirect.rs"]
mod indirect;
#[path = "core_layer_blit.rs"]
mod layer_blit;
#[path = "core_mapping.rs"]
mod mapping;
#[path = "core_mrt_indexed.rs"]
mod mrt_indexed;
#[path = "core_mrt_resolve.rs"]
mod mrt_resolve;
#[path = "core_mrt_surface_resolve.rs"]
mod mrt_surface_resolve;
#[cfg(any(feature = "native-gl-wgl", feature = "native-gles-egl"))]
#[path = "core_msaa_array_formats.rs"]
mod msaa_array_formats;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_msaa_array_sampling.rs"]
mod msaa_array_sampling;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_msaa_copy.rs"]
mod msaa_copy;
#[path = "core_multiview.rs"]
mod multiview;
#[cfg(any(
    all(windows, feature = "native-gl-wgl"),
    feature = "native-gles-egl",
    all(target_arch = "wasm32", feature = "webgl2")
))]
#[path = "core_multiview_timestamp.rs"]
mod multiview_timestamp;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_native_color_readback.rs"]
mod native_color_readback;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_native_integer_copy.rs"]
mod native_integer_copy;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_native_interop.rs"]
mod native_interop;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_native_mixed_srgb.rs"]
mod native_mixed_srgb;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_native_srgb.rs"]
mod native_srgb;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_native_surface_features.rs"]
mod native_surface_features;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_native_view.rs"]
mod native_view;
#[path = "core_ordinary_clear.rs"]
mod ordinary_clear;
#[path = "core_ordinary_formats.rs"]
mod ordinary_formats;
#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_packed_plane_copy.rs"]
mod packed_plane_copy;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_pbo_readback.rs"]
mod pbo_readback;
#[path = "core_pbo_upload.rs"]
mod pbo_upload;
#[path = "core_pipeline_override.rs"]
mod pipeline_override;
#[path = "core_pipeline_statistics.rs"]
mod pipeline_statistics;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_program_cache.rs"]
mod program_cache;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_push_compute.rs"]
mod push_compute;
#[path = "core_push_constants.rs"]
mod push_constants;
#[path = "core_push_matrix.rs"]
mod push_matrix;
#[cfg(feature = "native-gles-egl")]
#[path = "core_query_stencil_extract.rs"]
mod query_stencil_extract;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_raster_modes.rs"]
mod raster_modes;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_rgb9e5_readback.rs"]
mod rgb9e5_readback;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_sample_mask.rs"]
mod sample_mask;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_sampler_border.rs"]
mod sampler_border;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_sampler_border_integer.rs"]
mod sampler_border_integer;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_snorm_endpoint_readback.rs"]
mod snorm_endpoint_readback;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_srgb_surface.rs"]
mod srgb_surface;
#[path = "core_stencil_blit.rs"]
mod stencil_blit;
#[cfg(feature = "native-gles-egl")]
#[path = "core_stencil_extract_mask.rs"]
mod stencil_extract_mask;
#[cfg(feature = "native-gles-egl")]
#[path = "core_stencil_extract_mrt.rs"]
mod stencil_extract_mrt;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_stencil_transfer.rs"]
mod stencil_transfer;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_storage_3d.rs"]
mod storage_3d;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_storage_array.rs"]
mod storage_array;
#[cfg(any(all(windows, feature = "native-gl-wgl"), feature = "native-gles-egl"))]
#[path = "core_storage_formats.rs"]
mod storage_formats;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_surface_features.rs"]
mod surface_features;
#[path = "core_timestamp.rs"]
mod timestamp;
#[path = "core_transition_validation.rs"]
mod transition_validation;
#[path = "core_vertex_formats.rs"]
mod vertex_formats;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "webgl2_clear_masks.rs"]
mod webgl2_clear_masks;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "webgl2_raster_features.rs"]
mod webgl2_raster_features;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_d16_exhaustive.rs"]
mod webgl_d16_exhaustive;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_depth16.rs"]
mod webgl_depth16;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_depth16_array.rs"]
mod webgl_depth16_array;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_depth32.rs"]
mod webgl_depth32;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_integer_copy.rs"]
mod webgl_integer_copy;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_multisampled_sampled_reject.rs"]
mod webgl_multisampled_sampled_reject;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_pbo_readback.rs"]
mod webgl_pbo_readback;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_query.rs"]
mod webgl_query;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_raster_modes.rs"]
mod webgl_raster_modes;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_stencil_array.rs"]
mod webgl_stencil_array;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_stencil_depth_preserve.rs"]
mod webgl_stencil_depth_preserve;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_stencil_transfer.rs"]
mod webgl_stencil_transfer;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_unorm_feedback.rs"]
mod webgl_unorm_feedback;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_unorm_readback.rs"]
mod webgl_unorm_readback;
#[cfg(all(target_arch = "wasm32", feature = "webgl2"))]
#[path = "core_webgl_zero_stride.rs"]
mod webgl_zero_stride;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_direct_shader_artifacts.rs"]
mod direct_shader_artifacts;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_direct_spirv_storage.rs"]
mod direct_spirv_storage;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_direct_spirv_pairs.rs"]
mod direct_spirv_pairs;

#[cfg(all(windows, feature = "native-gl-wgl"))]
#[path = "core_direct_spirv_depth.rs"]
mod direct_spirv_depth;
