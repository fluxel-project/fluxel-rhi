# DX12 feature fixture index

This index maps public DX12 paths to targeted fixtures. It is not a frozen
candidate acceptance report: the final commit, adapter, driver, run results,
and representative presentation evidence must be recorded after the final
candidate is selected. Capability-gated tests can skip on an unsupported
device; a successful test process alone does not establish that a skipped path
executed.

| Area | Targeted fixtures | Scope |
| --- | --- | --- |
| Raster attachments | `mrt_gpu`, `stencil_gpu`, `readonly_depth_stencil_gpu`, `r32_float_raster_gpu` | MRT, blend, depth/stencil, read-only planes, and selected color/depth formats and resolves |
| Primitives and inputs | `primitive_gpu`, `vertex_formats_gpu` | Selected topologies, wireframe, strip restart, and selected vertex formats |
| Multiview | `multiview_mask_gpu`, `multiview_depth_gpu`, `multiview_resolve_gpu` | One to four views, sparse masks, secondary commands, depth clear, and MSAA resolve |
| Secondary | `secondary_scissor_gpu`, `secondary_restore_gpu`, `secondary_group_restore_gpu`, `secondary_vertex_restore_gpu`, `secondary_lifetime_gpu` | Bundle segmentation, resource lifetime, and primary state restoration |
| Compute and indirect | `compute_indirect_gpu`, `multi_count_gpu` | Indirect dispatch and GPU-count drawing |
| Dynamic bindings and arrays | `dynamic_storage_gpu`, `dynamic_structured_gpu`, `dynamic_storage_raster_gpu`, `descriptor_arrays_gpu`, `descriptor_texture_arrays_gpu`, `descriptor_uniform_arrays_gpu` | Selected bounded storage offsets, descriptor arrays, and invalid binding cases |
| Push constants | `push_ranges_gpu`, `push_stages_gpu` | Ranges and stage visibility |
| WGSL | `wgsl_raster_gpu`, `wgsl_vertex_index_gpu` | Selected raster WGSL translation, specialization, and vertex/instance indices |
| Pipeline cache | `pipeline_cache_gpu`, `compute_cache_gpu`, `mesh_gpu` | Native PSO cache serialization/reopen, malformed data rejection, and selected GPU output |
| Ray and mesh | `ray_dynamic_gpu`, `ray_query_gpu`, `mesh_gpu` | Raygen dynamic bindings, inline query hit/miss, and selected mesh outputs; this is not an exhaustive ray shader-stage matrix |
| Sparse, synchronization and queries | `sparse_gpu`, `sparse_switch_gpu`, `sparse_rt_ds_gpu`, `multi_queue_gpu`, `ownership_buffer_gpu`, `ownership_texture_gpu`, `query_gpu`, `query_cross_queue_reuse_gpu` | Selected mapping, queue, ownership, and query paths |
| Texture formats and copies | `bc_texture_gpu`, `bc_mip_gpu`, `bc_unaligned_gpu`, `bc6_bc7_sample_gpu`, `texture_upload_gpu`, `texture_d1_gpu`, `texture_d3_gpu`, `depth_stencil_plane_copy_gpu` | Selected BC formats and blocks, uploads, texture dimensions, and depth/stencil planes |
| Clears and presentation | `integer_clear_gpu`, `integer_clear_layers_gpu`, `color_rt_clear_gpu`, `clear_gpu`, `surface_gpu` | Selected standalone and pass clears; surface test is a presentation smoke test |

Additional targeted fixtures in the current tree:

| Area | Fixtures and boundary |
| --- | --- |
| Depth mip and array copies | `depth_mip_gpu`, `depth_array_copy_gpu`, `color_array_copy_gpu`, `d1_array_copy_gpu`, `bc_array_copy_gpu`; selected subresources and formats |
| Query packet wait | `query_cross_queue_reuse_gpu` exercises same-packet GPU-wait ordering and cross-packet transfer with a pending producer and native GPU wait |
| Upload and secondary formats | `upload_storage_gpu`, `upload_arena_gpu`, `secondary_formats_gpu`; selected storage and attachment cases |
| Presentation formats | `surface_formats_gpu` covers selected float and 10-bit SDR presentation formats as a smoke test |
| WGSL resources | `wgsl_volume_uav_gpu`, `wgsl_volume_sample_gpu`, `wgsl_uint_uav_gpu`, `wgsl_msaa_sample_gpu`, `wgsl_ray_query_gpu`; selected typed bindings and device-gated ray queries |
| Blit | `blit_gpu`, `blit_d1_gpu`, `blit_d3_gpu`, `blit_formats_layers_gpu`, `blit_bc_source_gpu`, `blit_compute_restore_gpu`, `blit_ray_restore_gpu`, `blit_buffer_formats_gpu`; selected compute/graphics, D1/D3 and array layers, formats and state restoration; raw-buffer fallback covers 43 ordinary color formats |

Fixtures live in [`crates/dx12/tests`](../../../crates/dx12/tests). Run an
individual target on Windows with `cargo test -p fluxel-rhi-dx12 --features
hardware-tests --test <fixture> -- --nocapture`. Set `FLUXEL_DX12_DEBUG=1`
for D3D12 debug-layer diagnostics. The final acceptance run should record the
exact invocation and whether each capability-gated case executed.

These targeted results remain subject to frozen-candidate gates. They do not define cross-numeric or mixed-dimension blit semantics, or establish support for every format and descriptor combination.
