# Vulkan completeness fixture index

This index identifies source fixtures and contract boundaries; their presence
does not establish that they passed on a particular device. The
[acceptance record](acceptance/README.md) identifies each tested source SHA,
commands, GPU and driver, output, diagnostics and known skips. Existing
[0.17 Vulkan evidence](../rhi-advanced-0.17/README.md) belongs to its own
recorded candidate and does not establish results for this working tree.

| Area | Fixture or implementation source | Scope |
| --- | --- | --- |
| Resource descriptors and compression | [resources_gpu.rs](../../../crates/vulkan/tests/resources_gpu.rs), [texture_extended.rs](../../../crates/vulkan/tests/hardware/texture_extended.rs) | Device-checked texture creation; BC1 array block copy and readback. The creation fixture skips unsupported formats. |
| Copies, clears and resolve | [hardware.rs](../../../crates/vulkan/tests/hardware.rs), [texture_extended.rs](../../../crates/vulkan/tests/hardware/texture_extended.rs), [texture_planes.rs](../../../crates/vulkan/tests/hardware/texture_planes.rs), [clear_gpu.rs](../../../crates/vulkan/tests/clear_gpu.rs) | Selected buffer and texture copies, subresources, clear, blit and multisample resolve. |
| Mapping and placement | [mapping_gpu.rs](../../../crates/vulkan/tests/mapping_gpu.rs), [placed_texture_gpu.rs](../../../crates/vulkan/tests/placed_texture_gpu.rs) | Mapping lease and exclusion; placed buffer offsets and placed texture lifetime/readback. |
| Native interop | [interop.rs](../../../crates/vulkan/src/backend/interop.rs) | Recording a native fill followed by RHI readback; device guard lifetime. Core's compile-fail example checks the encoder borrow. |
| External resources | [external_gpu.rs](../../../crates/vulkan/tests/external_gpu.rs), [external_texture_gpu.rs](../../../crates/vulkan/tests/external_texture_gpu.rs) | Windows opaque handle and semaphore buffer/texture fixtures, conditional on external device support. |
| Graphics and bindings | [mrt_gpu.rs](../../../crates/vulkan/tests/mrt_gpu.rs), [stencil_gpu.rs](../../../crates/vulkan/tests/stencil_gpu.rs), [readonly_depth_stencil_gpu.rs](../../../crates/vulkan/tests/readonly_depth_stencil_gpu.rs), [wgsl_sampler_array_gpu.rs](../../../crates/vulkan/tests/wgsl_sampler_array_gpu.rs), [sampler_clamp_zero_gpu.rs](../../../crates/vulkan/tests/sampler_clamp_zero_gpu.rs), [push_stages_gpu.rs](../../../crates/vulkan/tests/push_stages_gpu.rs) | Selected attachments, depth/stencil, binding arrays, integer and float clamp-to-zero sampling, and stage-specific push constants. |
| Compute, queries and queues | [compute_features_gpu.rs](../../../crates/vulkan/tests/compute_features_gpu.rs), [query_gpu.rs](../../../crates/vulkan/tests/query_gpu.rs), [multi_queue_gpu.rs](../../../crates/vulkan/tests/multi_queue_gpu.rs), [multi_count_gpu.rs](../../../crates/vulkan/tests/multi_count_gpu.rs) | Selected compute, query reuse, cross-lane waits and GPU-count drawing. |
| Mesh, ray and sparse | [mesh_gpu.rs](../../../crates/vulkan/tests/mesh_gpu.rs), [mesh_multiview_gpu.rs](../../../crates/vulkan/tests/mesh_multiview_gpu.rs), [ray_query_gpu.rs](../../../crates/vulkan/tests/ray_query_gpu.rs), [sparse_switch_gpu.rs](../../../crates/vulkan/tests/sparse_switch_gpu.rs), [sparse_volume_gpu.rs](../../../crates/vulkan/tests/sparse_volume_gpu.rs) | Device-gated mesh, mesh multiview and ray-query work; sparse mapping, mip switches, array and volume probes. |
| Sampler layout variants | [sampler_layout_switch_gpu.rs](../../../crates/vulkan/tests/sampler_layout_switch_gpu.rs), [sampler_runtime_mixed_gpu.rs](../../../crates/vulkan/tests/sampler_runtime_mixed_gpu.rs) | Ordinary, mixed and integer sampler pipeline switches without rebinding the public group; dynamically indexed mixed runtime samplers. |
| Resume | [resume_gpu.rs](../../../crates/vulkan/tests/resume_gpu.rs) | Completed prefix and original-plan suffix, with End and Steps completion requests. |
| Presentation | [surface_gpu.rs](../../../crates/vulkan/tests/surface_gpu.rs), [unix_surface_gpu.rs](../../../crates/vulkan/tests/unix_surface_gpu.rs) | Windows GPU presentation and Linux Xlib/XCB software presentation. Wayland is implemented but has no runtime evidence here. |

## Boundaries requiring device or contract checks

| Boundary | Current condition and evidence limit |
| --- | --- |
| Compression | BC, ETC2/EAC and ASTC require their native compression feature and format/usage support. BC1 has a copy fixture. The current test device reports ETC2/EAC and ASTC unavailable; those paths have no GPU acceptance on that device. |
| Copy and blit | Image-type copies requiring maintenance5 need that feature; blit format capabilities and resolve queue support are checked per device. |
| Depth/stencil | Separate aspect layouts require the corresponding device feature. Format, aspect, sample and attachment support remain descriptor-specific. |
| Placed memory | Select a supported native memory type from requirements, obey alignment and size, and reject dedicated-only placement. The caller remains responsible for aliasing and synchronization. |
| External memory | OPAQUE_WIN32 on Windows and OPAQUE_FD on Unix are the implemented memory and binary semaphore handle types. Device extensions and per-resource memory import and semaphore import/export support are required. The bridge supplies matching allocation metadata, resource parameters, ownership transfers and synchronization. Windows fixtures do not establish Unix GPU coverage. |
| Native interop | Device and encoder handles are Vulkan dispatchable handles. Encoder access is limited by an exclusive Core borrow; the caller restores state, coordinates synchronization and retains native-command resources through completion. |
| Sparse | Residency, image dimensions, sample counts, aspect requirements, queue and timeline support are device-specific. Shared mip tails and aspect groups follow reported tiling. Sparse depth and multi-aspect need supporting-device GPU evidence. |
| Optional GPU features | Mesh/task, ray tracing, ray query, descriptor indexing, multiview, multi-draw count and independent queues depend on enabled features, extensions, limits and valid descriptors. A capability skip is not a passing GPU execution. |
| Query domains | Vulkan's eleven-field pipeline-statistics query is feature-gated. It does not implement Core's distinct pass-boundary statistics sampling domain. Timestamp availability and valid bits depend on the queue family. |
| Shader and cache | SPIR-V is the native route; WGSL vertex, fragment and compute entries are lowered to SPIR-V. Shader stages, features and binding forms remain subject to validation and device support. Pipeline cache data is checked against the device identity. |
| Standalone clear | Buffer fill requires transfer-destination usage and four-byte alignment. Texture clear uses attachment usage and a graphics lane with caller-supplied attachment transitions. |

## Core domain coverage

The source implements the deferred command, resource, shader, raster, compute,
completion, readback, surface, query and cache interfaces, together with the
optional secondary, mesh, ray, sparse, multi-queue, mapping, placement, external
and native interop interfaces. Each optional operation remains subject to its
reported capability, lane, descriptor and native feature checks. The immediate
execution interface belongs to immediate backends and is not a Vulkan route.
The separate pass-boundary statistics interface is not implemented by Vulkan;
its pipeline-statistics query is a different contract. Source coverage does not
replace candidate GPU results.

The acceptance record distinguishes earlier full gates from affected-path
reruns and the final example corrections; see it for the precise evidence.
