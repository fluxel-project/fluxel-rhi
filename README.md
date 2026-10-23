# Fluxel RHI

Fluxel RHI is the GPU execution layer consumed by Fluxel's RenderGraph. It creates native resources,
executes caller-planned work, presents surface images, and reports completion.
RenderGraph and Renderer own scheduling, hazard analysis, frame policy, and
resource retirement; RHI supplies the native mechanisms and observable results
that those decisions require.

Renderer integrations select a concrete backend at initialization. `Device<B>`, resources,
encoders, and recorded work retain that type. Device identity checks catch
cross-device use even when two devices use the same backend. Capabilities remain
specific to the actual device and requested descriptor.

The integration criterion is whether a compiled RenderGraph plan maps directly
to RHI execution without rebuilding dependencies or transferring scheduling and
retirement policy into RHI. Examples are conformance workloads, not a reason to
replace explicit steps, waits, lanes, or completion facts with application APIs.

## Workspace

| Crate | Responsibility |
| --- | --- |
| [`fluxel-rhi-core`](crates/core/src/api/mod.rs) | Shared API, resource identity, execution contracts, and backend SPI |
| [`fluxel-rhi`](crates/rhi/src/lib.rs) | Feature-selected reexports and numbered integration examples |
| [`fluxel-rhi-dx12`](crates/dx12/src/backend.rs) | Direct3D 12 resources, command lists, descriptors, fences, and presentation |
| [`fluxel-rhi-vulkan`](crates/vulkan/src/backend/mod.rs) | Vulkan resources, dynamic rendering, command buffers, synchronization, and WSI |
| [`fluxel-rhi-webgpu`](crates/webgpu/src/backend/mod.rs) | Browser WebGPU encoding, queue promises, and canvas surfaces |
| [`fluxel-rhi-gl`](crates/gl/src/backend/gl/mod.rs) | Shared GL semantics with desktop GL/WGL, GLES/EGL, and WebGL2 profiles |
| [`fluxel-rhi-metal`](crates/metal/src/backend.rs) | Metal resources, encoding, completion, and layer surfaces |

Core defines operation meanings shared by callers and implementations. The
facade adds no runtime backend dispatcher. Each backend owns native objects,
shader lowering, state adaptation, and execution lifetimes.

## Backend routes

| Route | Facade feature | Execution model | Runtime shader route |
| --- | --- | --- | --- |
| Direct3D 12 / Windows | `dx12` (default) | Deferred native command lists | DXIL, or WGSL lowered through Naga and DXC |
| Vulkan | `vulkan` | Deferred native command buffers | SPIR-V, or WGSL lowered through Naga |
| Browser WebGPU | `webgpu` | Deferred native command buffers | WGSL |
| Desktop GL / WGL | `native-gl-wgl` | Immediate context-thread execution | WGSL lowered to profile-specific GLSL |
| GLES / EGL | `native-gles-egl` | Immediate context-thread execution | WGSL lowered to profile-specific GLSL |
| Browser WebGL2 | `webgl2` | Immediate browser-thread execution | WGSL lowered to GLSL ES |
| Metal / Apple targets | `metal` | Deferred native command buffers | WGSL lowered to MSL, or Metal source |

## GPU feature support

The table describes the current **public RHI route**, not every feature of the
underlying graphics API. `Device::capabilities()`, `Device::queries()`, and
descriptor support checks decide whether an operation is available on a
particular device. A listed route can still reject unsupported formats,
topologies, shader stages, queue plans, or device tiers.

| Backend | Main implemented GPU features | Advanced feature limits |
| --- | --- | --- |
| DX12 | Graphics and compute, multiple render targets, indirect drawing and dispatch, queries, secondary recording, texture clears, and pipeline caches | Device-gated descriptor indexing, up to four multiview views, ray pipelines and inline ray queries, mesh shaders, sparse textures, graphics/compute queues, and GPU-count multi-draw |
| Vulkan | Graphics and compute, multiple render targets, indirect drawing and dispatch, queries, secondary recording, clears, pipeline caches, mapping, placed memory, and native interop | Descriptor indexing, multiview, ray pipelines and inline ray queries, mesh shaders, sparse textures, graphics/compute queues, GPU-count multi-draw, and external resources depend on device features, extensions, formats and queue support. |
| Metal | Graphics and compute, up to eight render targets, depth/stencil, dynamic uniform/storage offsets, indirect draw and dispatch, standalone clears, copies/resolves, mapping, placement heaps, binary archives, and native interop | Device-gated argument arrays/indexing, mesh/object shaders, acceleration structures and ray queries, multiview, timestamp/occlusion and separate pass counters, cross-packet graphics/compute synchronization, GPU-count multi-draw, placement sparse textures, and external resources. No Metal tests or GPU acceptance were run in this phase, as requested. |
| Browser WebGPU | Graphics and compute, MRT, depth/stencil, dynamic offsets, indirect draw/dispatch, multi-draw including GPU-count lowering, standalone clears, copies/blits/resolves, buffer mapping and readback, occlusion and optional timestamp queries | Optional native features and limits gate compressed formats, immediates, subgroups, dual-source blending and unclipped depth. Ray/mesh, sparse resources, native multiple queues, multiview, complete pipeline statistics, native heap placement and serialized pipeline caches have no matching standard browser route. |
| Desktop GL4+ | Immediate graphics, MRT, depth/stencil transfers, copies/resolves, bounded runtime binding arrays, GPU-staged readback, compute and indirect commands where supported | Native limits gate nonuniform indexing, persistent coherent mapping, GPU-count drawing, queries/statistics, program binary caches, conservative rasterization and contiguous OVR multiview. |
| GLES3+ | Immediate graphics, MRT, bounded runtime arrays, depth/stencil uploads and copies, GPU-staged readback; compute, nonuniform indexing and indirect commands where supported | ES3.1 and extensions gate GPU-count lowering, sampled depth/stencil readback, coherent persistent mapping, exact color readback, queries, program caches and contiguous OVR multiview. |
| Browser WebGL2 | Immediate graphics, MRT, bounded runtime binding arrays, depth/stencil plane transfers, copies/resolves, GPU-staged readback, asynchronous snapshot mapping, and occlusion | Native limits bound array specialization and exact format readback; the combined nonuniform texture/SSBO feature remains unavailable. Extensions gate compressed textures and other optional operations. Standard WebGL2 has no compute or compressed-block readback. |

Metal's separate pass-boundary sample API exposes up to eight native counter
fields with a device availability mask and an error sentinel. It returns raw
stage samples, which cannot be assumed subtractable or equivalent to complete
pipeline statistics. The existing eleven-field `PipelineStatistics` query
remains unsupported on Metal. This path has not run on an Apple GPU.

Vulkan exposes the separate eleven-field pipeline-statistics query when the
device supports it. It does not expose the separate pass-boundary statistics
sampling domain.

Vulkan external interop imports opaque Win32 handles on Windows and opaque file
descriptors on Unix, with binary semaphore synchronization and explicit queue
ownership transfers. The producer must supply matching resource and allocation
metadata, and device extensions and per-resource import support are required.
The Windows fixtures do not establish Unix GPU coverage.

Metal's GPU-count draw route probes the native conversion shader and both ICB
command types on the actual device before advertising support. It keeps the count and argument buffers on the GPU and uses the
active pipeline topology. The runtime probe does not establish visual or synchronization
correctness; no Apple GPU result is recorded yet.

Metal TLAS dependencies can be consumed while the build and dispatch are
recorded in one encoder. Another encoder can bind that TLAS only after the
build's completion succeeds; recording that consumer early returns
`InvalidState`. Dropping an unsubmitted build releases its dependencies and
allows a new recording. A failed submitted build cannot be reused.

WebGPU lowers GPU-count drawing into GPU-generated indirect arguments and
native pass segments, without reading the count on the CPU. Push constants use
native immediate data only when the browser exposes the required feature and
limit. Compressed formats remain optional adapter features; the current AMD
browser evidence covers BC1, not ETC2/EAC or ASTC. Browser examples may select
their existing explicit fallbacks for
descriptor indexing and multiview; those scenes do not establish backend support
for either feature.

### DX12 pipeline cache

DX12 pipeline caches persist native cached PSO blobs for raster, compute, and
mesh pipelines. The cache verifies its serialized data and retries PSO creation
without a cached blob when the adapter or driver is incompatible. DX12 ray
pipelines use state objects rather than PSOs and reject a supplied cache;
Direct3D 12 exposes [cached PSO blobs](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_cached_pipeline_state); ray tracing instead uses [state objects](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist4-setpipelinestate1).

### DX12 graphics and shader route

The following summarizes implemented public paths. Format, shader-model, tier,
and resource descriptor checks still apply to each device; a capability does
not imply that every combination is supported.

| Area | DX12 public route and limits |
| --- | --- |
| Raster | Up to eight color attachments, per-target blend and write masks, depth/stencil, read-only depth/stencil views, and attachment clears and resolves. Supported sample counts and formats remain device-dependent. |
| Compute and indirect | Compute dispatch, indirect dispatch, indirect draw, and GPU-count multi-draw. Dynamic uniform and bounded raw/structured storage offsets are validated and retained through completion. |
| Resources | Buffer memory classes, texture upload/copy/readback, including array and depth mip paths, sampling and storage views, descriptor arrays, and standalone clears. Query format support for the actual resource and usage. |
| Bindings | Descriptor arrays and indexing are gated by device capabilities and descriptor restrictions. Push constants have stage-specific visibility; dynamic offsets apply to the full bound range. |
| Secondary and multiview | Secondary bundles preserve primary graphics state; supported view instancing tiers expose up to four views with view masks. |
| Advanced | Acceleration structures, ray pipelines, ray queries, mesh shaders, sparse resources, queries, and multiple queues require the corresponding reported capability and supported descriptor. |

DX12 accepts native DXIL and WGSL vertex, fragment, and compute entries. WGSL is
validated and lowered to HLSL by Naga, then compiled with the DXC runtime; DXC
and the required shader model must be available. Shader resources, overrides,
stages, and binding forms are subject to the supported lowering subset. Selected
WGSL compute inline ray queries are supported when the device exposes the required
ray-query capability; ray pipeline and mesh shader stages use native DXIL. WGSL support does not imply
that every WGSL feature or shader is accepted.

DX12 also supports device-checked float and 10-bit SDR surface formats and
selected compute/graphics blits across D1, D3 and array layers, including sRGB,
integer and BC source paths. WGSL resource lowering covers selected D3, storage,
multisampled textures and ray queries with binding type validation. Query slots can transfer across lanes with an explicit GPU wait, including a
pending producer in a different packet. The raw-buffer blit fallback has targeted
GPU coverage for 43 ordinary color formats. These are selected paths; cross-numeric
and mixed-dimension blit semantics are not defined by this coverage.

The DX12 candidate passed its complete hardware suite and final affected tests.
Corrected validation diagnostics, performance screening and visual scope were
reviewed; the raw run logs and captures are retained locally.

### Compressed texture formats

DX12 exposes the Core BC1–BC7 formats through device-checked texture creation,
block-aligned upload/copy, and sampling paths. Individual formats, dimensions,
and usages remain subject to `texture_support`. ETC2/EAC and ASTC have no DXGI
format mapping and are unsupported by the DX12 route. Vulkan maps these Core
families subject to the device compression features and per-format image usage
checks. The current Vulkan GPU fixture covers BC1 block copies; ETC2/EAC and
ASTC require a supporting device for GPU validation.

| Compressed family | DX12 | Vulkan | Metal | WebGPU | GL / GLES | WebGL2 |
| --- | --- | --- | --- | --- | --- | --- |
| BC1–BC7 | Device-checked public route | Device-checked public route; GPU validation is format/device dependent | Device-checked public route; not GPU tested | Optional standard feature; BC1 copy GPU checked | Core/extension-gated upload and sampling; `COPY_SRC` depends on the format and transfer route | Extension-gated upload and sampling; no compressed `COPY_SRC` |
| ETC2 / EAC | No DXGI mapping | Device-checked public route; GPU validation is format/device dependent | Apple-family public route; not GPU tested | Optional standard feature; GPU validation pending | Core/extension-gated upload and sampling; `COPY_SRC` depends on the format and transfer route | Extension-gated upload and sampling; no compressed `COPY_SRC` |
| ASTC | No DXGI mapping | Device-checked public route; GPU validation is format/device dependent | Apple-family LDR/HDR routes; not GPU tested | Optional LDR feature; GPU validation pending; no HDR mapping | Core/extension-gated LDR/HDR upload and sampling; `COPY_SRC` depends on the format and transfer route | Extension/profile-gated LDR/HDR upload and sampling; no compressed `COPY_SRC` |

For GL-family compressed textures, query `texture_support` for the exact
format, dimension, and usage. Compressed `COPY_SRC` depends on native image-copy
and readback entry points; GLES also requires a compatible copy-image route and
a probed integer attachment/readback path. GLES additionally probes the actual
compressed-to-integer block copy per format; a driver may expose image-copy
extensions while rejecting that particular reinterpretation. Ordinary
texture-to-texture copies retain their independent support. WebGL2 has no standard raw compressed
block readback. The WGL AMD fixture checked raw blocks for 11 representative
formats, and the GLES ES 3.2 run used Mesa llvmpipe. ASTC HDR was skipped on
those devices. Windows ANGLE ES 3.1 skips compressed readback when the required
copy-image extension is absent.

Desktop GL also supports BC6H/BC7 3D textures as independently compressed
2D slices, including CPU/PBO uploads, image copies, sampling, and raw block
readback. The WGL AMD fixture covers all four BPTC formats with two mip
levels and both immutable and mutable allocation. Desktop ASTC 3D additionally
requires the acquired KHR HDR extension, or KHR LDR plus sliced-3D support for
LDR/sRGB formats; these routes have no positive local GPU validation. S3TC,
RGTC, and ETC2/EAC 3D targets are rejected according to the desktop GL contract.

Desktop GL buffer readback also covers renderable normalized and floating-point
color formats in their native transfer encoding. A WGL fixture checks 19 formats
with finite representative values, subregions, mip/array selection, unaligned
offsets, row padding, and restoration of incoming read-color clamping.
Non-renderable RGB9E5 uses compatible GPU image copies through R32UI for
buffer readback, gated by image-copy/view compatibility and exact integer
readback support. Its WGL fixture includes equivalent shared-exponent encodings.
Ordinary desktop color formats also have framebuffer-independent native PBO
readback: GL4.5 subimage reads, or whole-mip `GetTexImage` followed by GPU region
copies on older contexts. Forty formats and 1D/array/3D subregions have focused
GPU coverage. The legacy entry was exercised on the current GL4.6 driver;
this does not establish coverage of a separate older driver.

Desktop GL4+ supports bounded runtime binding arrays through layout-capacity
specialization. Nonuniform sampled-texture and storage-buffer indexing requires
storage-buffer support. Native slot limits and texture/sampler pair expansion
still bound each pipeline; divergent implicit samples can increase sampling
cost. Public Core GPU fixtures cover UBOs, SSBO atomics, storage images, sampled
textures/samplers, short prefixes, and permitted holes.

GL4.4 or `ARB_buffer_storage` provides persistent coherent mapping for mappable
buffers. Buffer readback uses GPU staging; legacy query resolve checks
availability after explicit completion without waiting. Desktop depth/stencil
transfers include D16/D32/S8 and selected combined-format planes, preserving the
other plane; D24 depth has no portable byte encoding. PBO stencil packing needs
GL4.3 compute. Desktop OVR multiview supports contiguous view intervals, with
texture-view aliases required for nonzero attachment bases. Arbitrary masks,
multiview timer queries, and compressed color clears have no supported native
route; see the [desktop GL contract](documents/design-rhi.md#desktop-gl4--wgl).

GLES runtime binding arrays use the same bounded specialization, and nonuniform
indexing requires storage-buffer support. ES3.1 depth/stencil buffer copies use
GPU texture fetches and byte packing; standalone stencil textures require ES3.2
or `OES_texture_stencil8`. Exact ordinary-color readback uses validated native
format/type pairs or compatible integer image copies with the required
extensions. Focused GPU checks include SNORM endpoint encodings and floating
negative zero, NaN payloads and subnormals. `EXT_buffer_storage` provides coherent
persistent mappings on contexts that expose its actual entry point. The local
GLES matrix includes two existing ANGLE runtimes on the Radeon 780M; capabilities
and coverage differ between their D3D11 and Vulkan renderers.

GL implements `ImmediateBackend`; it does not expose deferred queue submission.
The GL family explicitly does not support secondary command recording,
independent queues, sparse/placed memory, external resources,
mesh shaders, ray pipelines, or ray queries. These domains are outside
its implementation scope, including driver-specific extension variants; see the
[GL support decisions](documents/design-rhi.md#unsupported-gl-family-domains).
Native desktop GL and EGL/GLES support scoped Core native interop with a borrowed
shared `NativeGlContext` dispatch table, exclusive owner-thread access, and state
invalidation on guard exit. WebGL2 exposes no native dispatch-table pointer.
WebGL2 does not provide native multithreaded command recording. WebGPU records
native commands even though browser objects belong to one thread. Separate
traits represent these different execution contracts.

For ordered Commands/Present plans, the optional statically dispatched
`Device::begin_execution` interface shares `commands`, `present`, and `finish`
across these models. GL calls native APIs inside `commands`; deferred backends
record native work and submit it at `finish`. No CPU command stream, replay,
per-step fence, or implicit idle wait is added. Borrowed `facts()` observations
share acceptance and completion evidence while retaining model-specific recovery
ownership. Failed-step facts distinguish no effect, partial issue, and unknown
effect; completion evidence does not turn a failed step into success.
Existing recording, submission, and immediate-session APIs remain
available for advanced execution. See the [execution contract](documents/design-rhi.md#common-execution-entry).

Support queries describe implemented native routes. An example may select an
explicit visual fallback for an unavailable feature; that does not enable the
feature in the backend. Metal has no completed Apple GPU or visual acceptance.
See the [conformance matrix](documents/conformance-matrix.md) and
[testing guide](documents/testing.md) for verification scope. A tested device
does not establish support for every device using that API.

## Build and run

On Windows with the Rust MSVC toolchain:

```powershell
cargo check --workspace --all-targets
cargo test -p fluxel-rhi-core --lib
cargo run -p fluxel-rhi --no-default-features --features examples,vulkan --example 01_triangle -- --backend vulkan --frames 2
```

The 23 [numbered examples](crates/rhi/examples/README.md) exercise scene
workloads through the public API. Examples 01–15 share the WGSL baseline;
16–23 exercise native advanced paths on DX12 and Vulkan. Their instructions
cover native backend
selection, browser entries, Android execution, and explicit feature fallbacks.
`--frames 2` ends the native run after two frames. Browser capture entries may
also intentionally stop after exporting a finite set of frames.

Some examples require [model and texture assets](crates/rhi/examples/assets/README.md).
[Example tools](tools/examples) fetch assets and generate native shader artifacts
from the maintained WGSL sources. Generated DXIL and SPIR-V do not imply runtime
WGSL support in those backends.

The [architecture and design rationale](documents/design-rhi.md) explains the
shared contracts, each crate's role, and the choices specific to each backend.
