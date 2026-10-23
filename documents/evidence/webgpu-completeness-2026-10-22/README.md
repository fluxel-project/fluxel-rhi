# WebGPU completeness acceptance

Implementation candidate: `907e67abb4cb39322973d47f7c6f8346bc9c666e`.
Acceptance ran on October 22, 2026. This evidence commit changes no backend or
example implementation. Metal was completed separately in `8bbff39` and
`9bffa24`; no Metal tests or Apple GPU acceptance were run, as requested.

## Results and provenance

| Gate | Result | Raw evidence |
| --- | --- | --- |
| Rust 1.87.0, wasm32, all WebGPU targets/features | Passed | [manifest](msrv-wasm.json), [stderr](msrv-wasm.stderr) |
| Stable Rust, wasm32, all WebGPU targets/features | Passed | [manifest](stable-wasm.json), [stderr](stable-wasm.stderr) |
| Rust 1.87.0, Windows MSVC, workspace all targets/features | Passed | [manifest](msrv-windows.json), [stderr](msrv-windows.stderr) |
| Browser tests, real GPU | 31 passed, 0 failed | [manifest](browser-tests.json), [output](browser-tests.stdout), [stderr](browser-tests.stderr) |
| Browser examples 01–15 | 15 completed; no browser diagnostics | [capture report](scenes/report.json) |
| PNG integrity, dimensions and nonblank output | 15 passed | [checker](check_images.py), [results](image-check.json) |

Every command manifest records the exact candidate and a clean working tree
before and after execution. Raw logs retain existing compiler warnings and the
wasm-bindgen initialization deprecation message. Zero test failures does not
mean warning-free builds.

The browser reports AMD RDNA 3 and `isFallbackAdapter=false`. Chrome was
155.0.8059.85 and ChromeDriver 155.0.8059.40 on Windows. The browser withholds
the detailed GPU/driver identity. [Host inventory](host-gpu.json) independently
reports Radeon 780M and driver 32.0.21028.2002; this is supporting host evidence,
not an identification field returned by WebGPU.

Tests used `wasm-bindgen-test-runner`, `WASM_BINDGEN_USE_BROWSER=1`,
`WASM_BINDGEN_TEST_TIMEOUT=180`, the compatible ChromeDriver, and the archived
[WebDriver configuration](webdriver.json). Captures used
[`webgpu_captures.py`](../../../tools/validation/webgpu_captures.py), headless
Chrome, `--enable-unsafe-webgpu`, `--disable-gpu-sandbox`, a 1280 by 720 canvas,
device scale 1, three frames and no injected input. The report retains generated
module hashes; generated wasm/JS files remain rebuildable output outside Git.

## Implemented routes and focused GPU checks

| Area | Current implementation and output evidence |
| --- | --- |
| Features and limits | Requests supported standard optional adapter features and limits; capability checks use the actual device. Chromium-private features visible in the adapter log are not enabled by the default feature list. |
| Buffers | Arbitrary byte uploads/copies and cropped readback, preserving neighboring bytes. Mapping uses mapAsync, exclusive ownership and owned Rust bytes. Fixtures cover odd offsets, 1031-byte copies and mapping exclusion. |
| Textures | Standard Core-mapped color/depth/stencil and optional BC/ETC2/EAC/ASTC LDR formats; dimensions, views, usage and regions checked before native calls. Odd buffer offsets and row strides use GPU repacking. |
| Depth/stencil transfer | Depth32 upload and filtered depth blit; single-plane combined depth/stencil copies preserve the other plane. Fixtures cover two array layers and distinct depth/stencil values in each of four MSAA samples. |
| Clear/blit/resolve | Standalone clears including 1D and ordinary nonrenderable formats; integer nearest, float linear, volume and depth blits; partial clear/resolve preserves pixels outside the render area. Fixtures include RGB9E5 encoding and partial MSAA resolve. |
| Compute | Specialization constants, dynamic bindings, indirect dispatch, native immediate data and optional subgroup size controls. Output checks include overrides plus indirect dispatch and the selected subgroup size. |
| Raster and immediates | MRT, depth/stencil, dynamic offsets, blend/stencil state, optional dual-source blending and unclipped depth. Immediate WGSL lowering covers selected-entry layouts, arrays, explicit alignment, pointer helpers and stage-specific values. Fixtures cover separate vertex/fragment values and scalar/array compute readback. |
| Indirect | Native indirect draw and CPU-count loops; GPU-count drawing snapshots count once, generates masked arguments on the GPU in bounded batches and restores native raster state. Fixtures cover count zero, one, clamping, indexed 20-byte arguments and 24-byte stride, attachment preservation and queries spanning native segments. No CPU count readback or draw replay is used. |
| Queries | Standard timestampWrites in nanoseconds; native occlusion segments are reduced on the GPU. Fixtures cover logical-slot reuse, sample-count changes, GPU-count pass splits and 4097 repeated queries across the native physical-slot boundary. Pass timestamps include auxiliary clear/resolve work. |
| Submission and surfaces | Queue promise completion retains native resources, including suspended canvas suffixes. Canvas configuration maps the standard three formats, five usage flags, two alpha modes and sRGB/DisplayP3; GPU validation covers each and combined usages. Host lifecycle remains outside RHI. |

BC1 upload/copy was exercised on this device. ETC2 and ASTC cases explicitly
skipped because the adapter does not expose their compression features. These
formats have implementation and feature gates, not GPU acceptance on this host.
No claim is made that every shader-language construct or format/usage combination
has a dedicated GPU fixture. Immediate lowering uses pinned Naga 30.0.1 and the
browser must accept the generated selected-entry WGSL.

## Scene review

All 15 original PNGs were opened by the integrating reviewer. Expected content
was present: the RGB triangle, textured panel, asteroid ring, colored cubes,
colored sphere ring, original/filtered compute images, reflected dragon,
spacecraft, mipmapped tunnel, vegetation, occluder and teapot, dragon capture,
UFO field, patterned cubes, and paired room views. The image checker additionally
checks CRCs, capture hashes, dimensions, output state and nonblank color coverage.

An independent reviewer opened examples 07, 10 and 15 and checked the manifest,
all PNG hashes, diagnostics and adapter identity. The reflected dragon, dense
vegetation and paired room scenes were nonblank and consistent with their
workloads. No scene source or configuration was modified for this acceptance.

These are static third-frame observations on one browser/GPU, not exhaustive
animation or long-duration acceptance. Example 05 retains its existing explicit
binding fallback; immediate-data support is established by focused fixtures.
Examples 14 and 15 retain their existing descriptor-indexing and multiview
fallbacks. Example 13 does not establish native cross-thread browser recording.
Examples 16–23 have no browser exports and were not run as WebGPU scenes.

## Remaining native and public-contract boundaries

| Boundary | Reason for retaining rejection |
| --- | --- |
| Ray pipelines/queries, mesh, sparse resources, native multi-queue synchronization, multiview, complete pipeline statistics | No corresponding standard browser WebGPU interface for these Core operations. Software ray tracing or renderer-generated multiple views would be separate rendering paths. |
| Binding arrays/indexing, border-color sampling, non-fill/conservative raster | No matching standard binding-array, sampler or raster descriptor; Chromium-private resource-table extensions are not substituted. |
| Secondary recording | Native render bundles cannot express the independent viewport/scissor operations of the current Core secondary contract. A CPU replay stream is not introduced. |
| Native heap placement, external OS handles/pointers, pipeline-cache serialization | Browser-managed resources expose none of these native ownership/storage interfaces. |
| Browser video external textures | WebGPU has GPUExternalTexture, but Core has no VideoFrame/HTMLVideoElement source and lifetime contract. Ordinary texture import is not equivalent. |
| Depth24Plus byte transfers | Core does not define a portable byte layout for this implementation-defined depth representation. Depth32 and stencil byte paths are implemented. |
| Compressed color clear or compressed blit destination | No native compressed render attachment or clear/encoding operation. A software BC/ETC/ASTC encoder is outside native backend lowering; this is not a claim that software encoding is impossible. |
| Unaligned partial writes into native MAP_READ destinations | Standard MAP_READ usage combinations prevent a GPU read of the old destination needed to preserve neighboring bytes. Aligned copy and read mapping remain available. |
| HDR10 PQ / ExtendedLinearSrgb canvas declarations | No matching standard native canvas colorSpace declaration; tone mapping alone does not establish the required transfer function. A renderer conversion path is separate. |
| Optional formats, timestamp, immediates and subgroup controls | Runtime features, language features and device limits must be present. ASTC HDR has no standard WebGPU texture-format mapping. |

The ownership and operation meanings remain those of
[`design-rhi.md`](../../design-rhi.md) and Core, checked against the
[WebGPU specification](https://gpuweb.github.io/gpuweb/) and
[WGSL specification](https://gpuweb.github.io/gpuweb/wgsl/).
Support is not inferred from native APIs on other backends.

## Closed implementation review findings

- WGPU-01: combined depth/stencil single-plane native copy is invalid. Replaced
  with per-layer/per-sample native render paths; single-plane, array and 4x GPU
  readback fixtures passed. An initial concern that array depth copies bypassed
  the layer loop was independently withdrawn after checking the call chain.
- WGPU-02: reopening a pass lost native binding state because an Array copy was
  not stored back. Corrected state retention; GPU-count/state/query fixture passed.
- WGPU-03: GPU-count scratch and dispatch limits restricted valid argument
  buffers. Lowering now batches against actual limits and shares one count snapshot.
- WGPU-04: occlusion physical-slot exhaustion rejected repeated logical queries.
  Native passes now split automatically; the 4097-query readback fixture passed.
- WGPU-05: helper clear/resolve work lay outside requested pass timestamps.
  Boundary timestamp writes now surround the full public pass lowering.

Independent static review checked pass store/load, read-only aspects, MSAA
resolve, batching, query aggregation, timestamps and completion retention. No
remaining confirmed defect was reported in that scope. The checks above establish
the stated implementation routes and host coverage, not universal device support.

All archived files are listed in [SHA256SUMS](SHA256SUMS). No release, tag or GL
backend work is part of this backend completion.
