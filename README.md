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
| [`fluxel-rhi-metal`](crates/metal/src/backend.rs) | In-progress Metal encoding, completion, and layer surface implementation |

Core defines operation meanings shared by callers and implementations. The
facade adds no runtime backend dispatcher. Each backend owns native objects,
shader lowering, state adaptation, and execution lifetimes.

## Backend routes

| Route | Facade feature | Execution model | Runtime shader route |
| --- | --- | --- | --- |
| Direct3D 12 / Windows | `dx12` (default) | Deferred native command lists | DXIL, or WGSL lowered through Naga and DXC |
| Vulkan | `vulkan` | Deferred native command buffers | SPIR-V |
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
| Vulkan | Graphics and compute, indirect drawing, occlusion/timestamp/pipeline-statistics queries, secondary recording, and device-gated descriptor indexing and multiview | Device-gated ray pipelines and inline ray queries, mesh shaders, sparse textures, graphics/compute queues, GPU-count multi-draw, and standalone texture clear |
| Metal | Graphics and compute, restricted standalone texture clear and compute storage buffers; device-gated mesh dispatch, triangle/AABB acceleration-structure builds, Apple-family timestamp sampling, pass-boundary counter samples, two native command queues, and triangle GPU-count multi-draw through ICBs | Compute storage buffers require a whole-buffer binding without dynamic offset. Multi-queue waits are limited to one packet and four waits per recording; cross-packet tokens are rejected. Ray queries, mesh task/object payload, DXR-style ray pipeline, exact eleven-field pipeline-statistics layout, and sparse mapping are not advertised. Apple-target compilation and static checks only; no Apple GPU runtime acceptance |
| Browser WebGPU | Graphics and compute, device-gated indirect drawing, and occlusion queries | The new ray, mesh, sparse, multi-queue, timestamp/statistics, GPU-count, and standalone-clear routes are not implemented here |
| Desktop GL / GLES | Immediate graphics; compute, indirect drawing, and occlusion depend on the actual context and profile | The new advanced routes are not implemented in the public GL backend |
| Browser WebGL2 | Immediate graphics through the browser owner | No native compute or new advanced route is exposed |

Metal's separate pass-boundary sample API exposes up to eight native counter
fields with a device availability mask and an error sentinel. It returns raw
stage samples, which cannot be assumed subtractable or equivalent to complete
pipeline statistics. The existing eleven-field `PipelineStatistics` query
remains unsupported on Metal. This path has not run on an Apple GPU.

Metal's GPU-count draw route probes the native conversion shader and both ICB
command types on the actual device before advertising support. It accepts the
current triangle-list raster subset and keeps the count and argument buffers
on the GPU. The runtime probe does not establish visual or synchronization
correctness; no Apple GPU result is recorded yet.

Metal TLAS dependencies can be consumed while the build and dispatch are
recorded in one encoder. Another encoder can bind that TLAS only after the
build's completion succeeds; recording that consumer early returns
`InvalidState`. Dropping an unsubmitted build releases its dependencies and
allows a new recording. A failed submitted build cannot be reused.

### DX12 pipeline cache

DX12 pipeline caches persist native cached PSO blobs for raster, compute, and
mesh pipelines. The cache verifies its serialized data and retries PSO creation
without a cached blob when the adapter or driver is incompatible. DX12 ray
pipelines use state objects rather than PSOs and reject a supplied cache;
Direct3D 12 exposes [cached PSO blobs](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_cached_pipeline_state); ray tracing instead uses [state objects](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist4-setpipelinestate1).

### DX12 graphics and shader route

The following summarizes implemented public paths. Format, shader-model, tier,
and resource descriptor checks still apply to each device; a capability does
not imply that every combination is supported. The [DX12 fixture index](documents/evidence/dx12-completeness-2026-10-22/README.md)
identifies the corresponding GPU tests and their scope.

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

The [DX12 review follow-up](documents/evidence/dx12-completeness-2026-10-22/review.md)
tracks the original review items and remaining acceptance work. The
[performance screening](documents/evidence/dx12-completeness-2026-10-22/performance.md)
contains the native-call measurements and their reproducibility limits. Targeted
GPU results remain subject to the frozen candidate acceptance gates.

### Compressed texture formats

DX12 exposes the Core BC1–BC7 formats through device-checked texture creation,
block-aligned upload/copy, and sampling paths. Individual formats, dimensions,
and usages remain subject to `texture_support`. ETC2/EAC and ASTC have no DXGI
format mapping and are unsupported by the DX12 route. Other backends currently
do not expose a complete public compressed-texture route.

| Compressed family | DX12 | Vulkan | Metal | WebGPU | GL / GLES | WebGL2 |
| --- | --- | --- | --- | --- | --- | --- |
| BC1–BC7 | Device-checked public route | No public route | No public route | No public route | No public route | No public route |
| ETC2 / EAC | No DXGI mapping | No public route | No public route | No public route | No public route | No public route |
| ASTC | No DXGI mapping | No public route | No public route | No public route | No public route | No public route |

GL implements `ImmediateBackend`; it does not expose deferred queue submission.
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
See the [advanced candidate evidence](documents/evidence/rhi-advanced-0.17/README.md),
[conformance matrix](documents/conformance-matrix.md),
[testing guide](documents/testing.md), and
[Android evidence](documents/evidence/android-2026-10-20/README.md) for the scope
of recorded verification. A tested device does not establish support for every
device using that API.

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
