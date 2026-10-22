# DX12 review follow-up

This maps the user-provided review to the current working tree and targeted
tests. These are implementation and screening results, not a final acceptance
for a frozen commit. Final candidate SHA, full affected GPU gates, independent
review, and presentation/visual evidence remain to be recorded. A fixture's
presence or a capability-gated skip does not by itself prove execution.

| Review item | Current response and evidence | Remaining acceptance |
| --- | --- | --- |
| P2-1: indirect resource states | Indirect draws use the caller-provided indirect state instead of silently transitioning from COMMON. `multi_count_gpu` covers GPU-generated indirect arguments, ordinary and count-based paths. | Final candidate rerun and review of the actual RenderGraph integration. |
| P2-2: signal per step | Consecutive same-lane steps can share native signaling while retaining per-step completion facts. See [performance screening](performance.md). | Final synchronization and failure-path review and candidate gates. |
| P2-3: upload resource per write | Upload staging is reused; the 500-write diagnostic checks all bytes and completion. See [performance screening](performance.md). | Final lifetime, concurrency and candidate gates, including cold allocation behavior. |
| P2-4: MRT | Multiple targets, per-target blend/write mask, clears, resolve and secondary inheritance are implemented. `mrt_gpu` and multiview fixtures cover selected combinations. | Final candidate gates; format/sample restrictions remain device-specific. |
| P2-4: compute dynamic offsets | Compute and ray use bounded dynamic CBV and storage descriptor lowering. `dynamic_storage_gpu`, `dynamic_structured_gpu`, and `ray_dynamic_gpu` cover selected values and invalid ranges. | Final candidate gates; descriptor and shader restrictions still apply. |
| P2-4: indirect dispatch | `compute_indirect_gpu` exercises indirect dispatch and readback. | Final candidate gates. |
| P2-4: blend and stencil | Per-target blend and depth/stencil lowering are covered by `mrt_gpu`, `stencil_gpu`, and `readonly_depth_stencil_gpu`. | Final candidate gates and device-specific format checks. |
| P2-4: buffer size limit | The fixed 1 GiB backend limit was replaced by device/operation checks; `buffer_limits_gpu` covers the selected limits and resource cases. | Final candidate gates; available GPU memory and native limits still constrain allocation. |
| P2-5: ShaderReadOnly stages | The transition lowering includes vertex, task, mesh and other non-pixel shader stages. | Unit coverage does not establish real-GPU execution for every stage combination; final affected tests and review remain. |
| Additional: per-list native execution | Compatible command lists are batched into native execution calls; the diagnostic counts 500 to 1 for the single-step workload. | Final ordering, retention and failure-path review. |
| Additional: repeated descriptor heaps | Repeated same-heap binding avoids redundant heap calls; the diagnostic counts 500 to 1. Timing improvement was within noise. | Final secondary/heap-switch and candidate correctness gates. |

Additional paths implemented during this work include DX12 PSO cache, ray
dynamic offsets, and up to four device-gated multiview views. The
[fixture index](README.md) states the scope of targeted coverage. Selected D1/D3 and multilayer blit and array-copy paths now have targeted
fixtures. Targeted GPU tests also pass for the raw-buffer blit fallback and
cross-packet query transfer with a pending producer and native GPU wait. Runtime support is constrained by the actual native device,
resource descriptor, and shader route.

## Subsequent DX12 coverage

The current tree adds targeted coverage for depth mips, array copies, upload
storage/SRV bindings, secondary attachment formats, float and 10-bit SDR
surfaces, and WGSL D3/storage/multisampled resources and inline ray queries.
`query_cross_queue_reuse_gpu` includes same-packet GPU-wait ordering. Selected
compute and graphics blits cover D1/D3, arrays, sRGB, integer and BC source
formats and state restoration. These are device- and descriptor-dependent
subsets; the [fixture index](README.md) identifies the tests.

The raw-buffer fallback passed targeted GPU tests for 43 ordinary color formats.
D1/D3 and array blits and compute/ray state restoration also passed targeted
tests. Cross-packet query transfer passed with a pending producer and native
GPU wait. Independent review has passed for submit, upload, descriptor heaps,
and the latest query work. These results remain subject to frozen-candidate
gates; they do not establish every format or descriptor combination, and
cross-numeric and mixed-dimension blit semantics remain undefined.

### DX12-LIFE-01: blit recording ownership

The compute blit now reserves every per-layer descriptor before recording its
first dispatch. It retains the output descriptor, staging resource, root
signature and pipeline before recording; each source descriptor is retained
before its dispatch. The graphics path retains each descriptor heap and
staging target before its draw, and `record::blit` retains both input textures
before recording either path. A later allocation or restoration error thus
cannot release objects already referenced by native commands.

`blit_failure_gpu` injects an out-of-memory result on the second layer and
checks the resulting sticky encoder failure. Core's `Encoder::call` poisons an
encoder after a backend error, so public callers cannot finish and submit it;
the original inference of a publicly triggerable GPU use-after-free was not
established. The ownership changes are retained as defense in depth. On the
local DX12 GPU, the injected failure fixture and the `blit_compute_restore_gpu`,
`blit_ray_restore_gpu`, `blit_buffer_formats_gpu`, and
`blit_formats_layers_gpu` fixtures passed. This result is bound to the working
tree and must be repeated or attributed to the frozen candidate separately.
