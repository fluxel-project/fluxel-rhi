# Open Design TODOs

This is the authoritative list of unfinished RHI work selected for follow-up.
An item stays open until its stated evidence and acceptance conditions are met.
Unselected capabilities below are not an automatic implementation backlog.

## Active items

### DX12 conservative rasterization

**Status:** Device capability reporting and graphics/mesh pipeline admission
use the D3D12 conservative rasterization feature and filled triangle rules.

**Acceptance:** Verify coverage on a supporting GPU. Check early rejection
for point/line topology, wireframe, and a device without support.

### DX12 surface attachment formats and HDR

**Status:** Configuration and acquisition retain the selected attachment view
format, including sRGB views of supported UNORM flip buffers.
DX12 also exposes current monitor color and luminance facts.
The HDR routes qualify scRGB and HDR10 against the current output and native
swapchain. Their real display acceptance is pending.

**Acceptance:** Reconfigure a live HWND between linear and sRGB views and
observe the format and generation on the acquired images. Exercise matching
raster pipelines and verify sRGB encoding on a real presentation path.
Compare output snapshots across a window move between distinct displays.
Verify HDR10 and scRGB output on an HDR-active monitor, early rejection of
incorrect format/color-space pairs and HDR on SDR, and reconfiguration after
an output move or switch to SDR.

### DX12 upload, completion, and texture transfer verification

**Status:** Direct staging writes, bounded completion waits, sampled color
component mapping, and descriptor-specific sample count reporting are wired.
The implementation has passed a Windows MSVC workspace development build.
Focused contract and GPU evidence are pending the implementation-stage report.

**Acceptance:** Read back a direct staging write and verify canceled writes
leave destination bytes and later command recording intact. Verify that a
pending completion times out and remains positive completion evidence only
after the native fence reaches it. Read back identity, rearranged, and constant
sampled components, and reject sampled-only views as storage or attachments.
Check the requested sample count and the reported alternatives against texture
creation for the same format, dimension, mip count, and usage.
Compare storage layout admission with actual typed UAV view creation for
matching formats and dimensions, and verify missing view roles fail before
descriptor copies.
Run direct color copies in both directions using texel-aligned offset and
pitch on a device that reports the feature. Check block alignment, array and
volume image strides, the exact final row, and rejection on the legacy path.

### DX12 comparison sampler integration

Scalar comparison samplers use the existing RHI layout and sampler descriptors.
The DX12 bind group checks the native sampler mode against the declared mode.
Rendering integration needs GPU evidence for a sampled depth view and a
comparison sampler in the same bind group.

**Acceptance:** Read back a depth-comparison result with distinct passing and
failing references, and verify that mismatched sampler modes are rejected.

### P7-LOWERING-PROFILE — expose cost-bearing lowering facts to RenderGraph

**Status:** Core exposes an optional per-device profile for GPU-count draws.
DX12 publishes its direct native route and its two vertex-index adjustment
routes. Route tests and classifications for the remaining backends are still
pending. WebGPU currently preprocesses GPU arguments, may allocate temporary
buffers, splits the pass, and emits repeated native draw calls. The P7 store-op
guard prevents one incorrect combination.

**Reopen/selection:** Keep scope limited to capabilities where a backend
lowering changes graph correctness, resource planning, synchronization, or
material CPU/GPU cost. Do not add a general-purpose cost model for all API
operations.

**Acceptance:** Define an immutable per-device lowering descriptor for the
selected cost-bearing operation(s), classify every implemented backend route
from source, and expose facts needed before graph compilation without adding
per-draw queries or allocation. Demonstrate that a caller can distinguish
native execution from GPU preprocessing, pass splitting, temporary resources,
and CPU iteration, then verify descriptors against backend route tests.

### P7-WEBGPU-SPLIT-STATE-COST — bound ordinary pass-state tracking overhead

**Status:** `pass_split::record` creates JavaScript objects/arrays and linearly
searches retained state for ordinary pipeline and binding updates, even when
the pass is never split. This is a concrete extra operation on the common path;
no profile currently proves its frame-time impact.

**Acceptance:** Add focused call/allocation counters or a representative CPU
profile that separates ordinary raster recording from split recovery. Only
then choose direct-indexed state or an explicit split-eligible pass contract;
verify unchanged state restoration after multiple splits.

### P7-RG-FRAMES-IN-FLIGHT — exercise cached plans with multiple live frames

**Status:** The example frame helper constructs a small plan per invocation and
waits for device idle after each frame. It is a smoke path and does not measure
cached-plan reuse, completion-frontier retirement, or overlapping frames.

**Acceptance:** Add one bounded RG-like execution workload that reuses a
compiled plan, submits 2–3 frames before waiting, keeps distinct completion
points, retires a recycled resource only after its covering completion, and
records CPU submit and GPU timings. Do not migrate all numbered examples to
this harness.

### P4-AS-ARRAY-VK — Vulkan fixed acceleration-structure arrays

**Status:** Unavailable; the bounded reconstruction failed and temporary
capability/admission changes were withdrawn. The distinct-TLAS result was
`[miss, hit, miss, miss]`, not the required `[hit, miss, hit, miss]`.

**Reopen when:** A concrete implementation fix, new source evidence, or a
relevant device/driver change is available. Do not restart the same fixture
without a discriminating change.

**Acceptance:** Validate the exact artifact and layout route before native
creation; prove four distinct-element hit/miss results on a real GPU; reject
unsupported shapes before native calls; and prove referenced acceleration
structures remain alive through completion after caller handles are dropped.

### P4-MESH-BATCH — Vulkan enhanced mesh batch/count fixture

**Status:** The enhanced fixture remains incomplete after repeated hangs. The
bounded static review found no source-proven correction. Existing verified
direct/indirect/task mesh routes remain as-is; the added count/sentinel claims
are not supported.

**Reopen when:** New fixture or runtime evidence identifies a concrete cause,
or the relevant target device/runtime changes. Do not repeat an unchanged
hanging run.

**Acceptance:** One bounded fixture must cover zero and boundary counts,
nonzero offsets, sentinel preservation, invalid usage/ranges, GPU-produced
arguments/counts, and in-flight resource retention, with distinguishable
real-GPU output and focused regression checks.

### P6-VIDEO-SAMPLING — host-owned video source integration

**Status:** The opaque `VideoFrame` API and WebGPU external-texture wrapper are
implemented. RHI does not decode video or own source creation, frame
scheduling, color conversion, or source closure. Real-source tests are deferred
to integration with `fluxel-host` as requested.

**Acceptance:** On a real browser GPU, separately import a `VideoFrame` and an
`HTMLVideoElement`; bind each through one fully-bound fragment external-video
entry and compare sampled readback against known frames with a documented
color-conversion tolerance. Drop public frame and bind-group handles after
recording and prove source roots remain alive through queue completion. Verify
that successive imports are distinct snapshots, that closing a `VideoFrame`
requires re-import, and that `HTMLVideoElement` import/encode/submit respects
browser task lifetime. Cover source failure, device loss, and cancellation
without sampling a stale frame. Keep source and decoder ownership in the host
adapter.

## Evidence boundary

Configured Surface lease invalidation at `b5d569f` has Core contract tests,
one real WebGPU canvas-resize refusal/reconfigure test, and Rendering's named
DX12/WGL presentation/resize evidence. Vulkan replacement-creation failure
after old-swapchain retirement and Metal out-of-band drawable fact mismatch
remain source/build evidence only. Reopen those native fault paths when a
reproducible failure-injection harness or Apple execution target is selected;
acceptance must show `SurfaceOutdated` and no old-generation lease publication.
See [the scoped validation record](documents/verification.md#configured-surface-lease-facts-2026-11-04).

Metal currently has no Apple GPU execution or Surface evidence in this
repository. This is not a claim of runtime acceptance; compile and source
review results must remain labeled accordingly. A backend capability claim
requiring GPU or presentation behavior needs real Apple-target evidence.

## Closed scope decisions

- **P7-WEBGPU-SPLIT-STORE:** User-owned writable attachments with `StoreOp::Discard`
  are rejected before internal pass termination. The backend-owned dummy depth
  attachment is exempt and cleared after every split. On commit `5866f9f`, the
  Chrome 156 browser suite passed the split-store cases (5/5) and the GPU-count
  readback/query case (1/1). Supporting logs are retained locally and are not
  part of the published source tree. The source is
  published as `fluxel-rhi-webgpu` 0.1.1; 0.1.0 remains immutable and predates
  this fix. Chrome did not expose physical GPU/driver identity, so closure is
  browser-route correctness evidence, not hardware-specific acceptance.
- Stage 4's selected route set and Stage 5 attachment-only allocation intent
  are closed. Closure does not claim every format, backend, array shape, tuple,
  or a physical memory/performance benefit.
- The fixed, fully-bound compute WGSL DX12 storage-buffer and uniform-buffer
  non-uniform routes are resolved. Runtime/partial arrays, dynamic offsets,
  non-compute stages, and mismatched shader/layout metadata remain unavailable.
- First-pass test-suite consolidation across the RHI crates is complete. The
  inventories and review record are in the local Temp archive; no backend
  capability change was part of that work.

## Deferred families, not active commitments

Reconsider these only when an owning consumer defines the required semantics
and supplies a bounded workload:

- VRS, Metal rasterization-rate maps, sampler feedback, and HDR output metadata.
- Vulkan local attachment reads or GLES framebuffer fetch; these require a
  RenderGraph tile-local access and ordering contract.
- Public shader compiler IR, portable error callbacks, allocation thresholds,
  automatic device-loss policy, and display-timing feedback.
- Geometry/tessellation stages, programmable sample locations, predication,
  and native video/planar formats or video sources beyond the current WebGPU
  wrapper.

Renderer algorithms, frame pacing, host/decoder lifecycle, cache eviction, and
asset residency remain owned above RHI.
