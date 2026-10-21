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
| Direct3D 12 / Windows | `dx12` (default) | Deferred native command lists | DXIL |
| Vulkan | `vulkan` | Deferred native command buffers | SPIR-V |
| Browser WebGPU | `webgpu` | Deferred native command buffers | WGSL |
| Desktop GL / WGL | `native-gl-wgl` | Immediate context-thread execution | WGSL lowered to profile-specific GLSL |
| GLES / EGL | `native-gles-egl` | Immediate context-thread execution | WGSL lowered to profile-specific GLSL |
| Browser WebGL2 | `webgl2` | Immediate browser-thread execution | WGSL lowered to GLSL ES |
| Metal / Apple targets | `metal` | Deferred native command buffers | WGSL lowered to MSL, or Metal source |

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
See the [conformance matrix](documents/conformance-matrix.md),
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

The 15 [numbered examples](crates/rhi/examples/README.md) exercise shared scene
workloads through the public API. Their instructions cover native backend
selection, browser entries, Android execution, and explicit feature fallbacks.
`--frames 2` ends the native run after two frames. Browser capture entries may
also intentionally stop after exporting a finite set of frames.

Some examples require [model and texture assets](crates/rhi/examples/assets/README.md).
[Example tools](tools/examples) fetch assets and generate native shader artifacts
from the maintained WGSL sources. Generated DXIL and SPIR-V do not imply runtime
WGSL support in those backends.

The [architecture and design rationale](documents/design-rhi.md) explains the
shared contracts, each crate's role, and the choices specific to each backend.
