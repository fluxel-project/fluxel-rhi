# Open Design TODOs

This is the authoritative list of unfinished RHI work selected for follow-up.
An item stays open until its stated evidence and acceptance conditions are met.
Unselected capabilities below are not an automatic implementation backlog.

## Active items

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
