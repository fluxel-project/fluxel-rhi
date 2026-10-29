# Open Design TODOs

This is the authoritative list of unfinished RHI work selected for follow-up.
An item stays open until its stated evidence and acceptance conditions are met.
Unselected capabilities below are not an automatic implementation backlog.

## Active items

### P7-WEBGPU-SPLIT-STORE — preserve discarded attachment semantics

**Status:** The unconditional split path has been guarded: internal WebGPU pass
splits reject any writable color, depth, or stencil attachment whose store op is
`discard`. This includes transient attachment routes and discarded MSAA sources
with resolve targets. The call fails before the native pass is ended. Targeted
Chrome 156 browser tests pass: the split-store validator rejects discarded
color/depth/stencil and allows stored/read-only cases; the GPU-count fixture
rejects a discarded color attachment and still reads back the expected pixels,
query result, and indexed-draw output on the stored-attachment path. The test
does not expose the physical adapter or driver selected by Chrome, so this is
browser WebGPU evidence, not identified hardware acceptance. The registry's
`fluxel-rhi-webgpu` 0.1.0 archive predates this source fix; the backend is being
released as 0.1.1 because crates.io versions cannot be replaced. A backend-
owned dummy depth attachment is separately exempted from the discard guard and
cleared when a split resumes; the focused Chrome regression now passes for an
attachment-free pass switching to a four-sample pipeline.

**Acceptance:** Compile the WebGPU conformance tests; run browser GPU regressions
proving discarded user attachments are rejected, the stored-attachment GPU-
count path still renders and reads back expected pixels, and an attachment-free
pass can split to a four-sample pipeline. Cover depth and stencil discard in
the split-store validator. Preserve the conservative rejection unless a
graph-visible split/store contract is implemented. Record physical GPU/driver
when the browser exposes it; do not claim hardware-specific acceptance from
the default browser adapter alone. Publish and verify `fluxel-rhi-webgpu`
0.1.1.

### P7-LOWERING-PROFILE — expose cost-bearing lowering facts to RenderGraph

**Status:** The Core capability model reports operation support but does not
describe important execution differences such as native GPU-count draw,
backend GPU preprocessing, internal pass split, temporary allocation, or CPU
iteration. WebGPU `multi_draw_count` is currently reported as supported even
though it preprocesses GPU arguments, may allocate temporary buffers, splits
the pass, and emits repeated native draw calls. The P7 store-op guard prevents
one incorrect combination but does not give graph compilation these cost facts.

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

Metal currently has no Apple GPU execution or Surface evidence in this
repository. This is not a claim of runtime acceptance; compile and source
review results must remain labeled accordingly. A backend capability claim
requiring GPU or presentation behavior needs real Apple-target evidence.

## Closed scope decisions

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
