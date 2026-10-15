# Fluxel RHI execution contract

**Status:** current implementation and design boundary.

## Public API and ownership

`fluxel_rhi_core::api` is the single Core API namespace. The facade re-exports it as
`fluxel_rhi::api`; applications use that namespace and select a concrete backend at
initialization. On Windows the facade exposes `fluxel_rhi::dx12::Dx12` and
`fluxel_rhi::vulkan::Vulkan`; browser WASM exposes `fluxel_rhi::webgpu::WebGpu`.
The GL-family route is exposed behind its WGL, EGL or WebGL2 feature. On Apple
targets, `fluxel_rhi::metal::Metal` is exposed behind the `metal` feature; this
backend is still in implementation and has no hardware conformance claim. Each
route has an optional feature.

The API is statically dispatched: `Device<B>`, resources, pipelines, encoders and
recorded work retain their concrete `Backend` type. RenderGraph or its host owns graph
scheduling, hazard analysis, lifetime policy and frame pacing. The RHI creates native
objects, issues or records the supplied operations according to the backend,
and reports support or failure. Deferred backends submit recorded native work;
GL reports work already issued through its immediate session.

A backend must keep public descriptors, support queries and command results aligned. A
feature is advertised only if the corresponding native route exists. Format- and
descriptor-specific limits belong in support queries or operation results. `Unsupported`
is the required result for a route that is absent.

## Graphics coordinate convention

The public raster contract follows the [WebGPU coordinate systems](https://www.w3.org/TR/webgpu/#coordinate-systems):
after perspective division, NDC X and Y are in `[-1, 1]`, NDC Y points up,
and NDC Z is in `[0, 1]`.
The framebuffer and viewport origin is the upper-left pixel, with X increasing
rightward and Y downward. `Viewport` and `Rect` use framebuffer coordinates;
`FrontFace` describes winding in framebuffer coordinates. For 2D textures,
texel `(0, 0)` is the upper-left texel and texel Y increases downward. Upload
row zero maps to that row; texture readback preserves this logical row order.
For sampled 2D textures, normalized `v = 0` addresses the upper edge and
`v = 1` the lower edge, including WGSL shaders lowered by a backend.

Backends preserve these observable semantics when lowering to native APIs.
DX12 already uses the matching native graphics convention. Vulkan uses a
negative-height viewport for the public framebuffer Y direction and maps
front-face winding in framebuffer coordinates.
Backends with a different convention must adapt viewport, winding, depth, and copy
lowering without changing the public contract. The RHI does not prescribe
world-space handedness, camera direction, matrix storage, or multiplication
order: those belong to the application's shader ABI and scene data. In
particular, matching a Vulkan example's screenshot must not introduce a
backend-wide Y flip into DX12.

## Current repository shape

- `crates/core/src/api/` defines the public vocabulary and backend SPI traits.
- `crates/rhi` is the typed facade and contains the numbered runnable examples.
- `crates/dx12` is an active native backend. Its lowering is organized under
  `backend/`, including `encoder/`, `pipeline`, `texture`, `descriptor`, `sampler`,
  `surface`, `query`, and `readback`.
- `crates/vulkan` is an active, still incomplete native backend with real device, resource,
  SPIR-V, Win32 WSI, command buffer, descriptor, pipeline, and fence-backed
  submission paths. All 15 numbered examples have Vulkan validation-layer
  runtime and GPU readback evidence.
- `crates/webgpu` is an active browser backend. It creates real `GPUDevice`
  resources, records native command/pass encoders, submits to `GPUQueue`, uses
  canvas presentation, and observes completion through browser promises.
- `crates/metal` is an in-progress native backend. Its current source has direct
  Metal command encoding, a limited resource/pipeline subset, completion
  observation, and a `CAMetalLayer` path. It has not passed macOS SDK linking,
  real Apple GPU execution, or example visual review; compile checks are not
  hardware evidence.
- `crates/gl` implements the current static Core API for the GL family. The WGL
  GL4+ route has individual real-GPU execution and visual evidence for all 15
  examples, including sequential fallbacks where native multiview, secondary
  recording, or runtime descriptor indexing is unavailable. The concentrated
  01–15 WGL run and final independent review are complete for the tested AMD
  Radeon 780M profile. EGL/GLES 3.1 has also passed all 15 examples on the same
  GPU through Mesa D3D12, with two-frame execution and visual comparison against
  the GL4+ and C++ references. Browser WebGL2 remains a separate profile and is
  not covered by either native route's evidence.

## Implemented DX12 baseline

The DX12 backend has real resource creation, descriptor/bind-group creation, DXIL shader
and raster-pipeline creation, command-list recording, graphics submission, surface
acquire/present, texture upload/copy routes, depth and color attachments, samplers,
readback, queries, mip generation, and raster indirect commands. Support remains
operation and descriptor dependent.

Examples exercise this baseline through `fluxel_rhi::api`. For example 10, the running
path loads the plant, ground and sky glTF assets; uploads the KTX mip chains and plant
texture array; encodes sky, ground and indexed-indirect plant passes; and presents the
surface image. Its DX12 route uses real `ExecuteIndirect` records.

Optional operations remain capability-gated. In particular, callers must check the
published indirect, multi-draw, push-constant, query, descriptor-indexing, multiview,
secondary-recording, and format support before selecting a workload. A compiling API
call is not evidence that every backend or descriptor can execute it.

Occlusion-query resolve values are unsigned integers. The portable contract is
that zero means no samples passed and any nonzero value means at least one
sample passed; the exact nonzero value is backend-defined. A backend may expose
the native sample count or a native boolean visibility result. Code that needs
only visibility must test for nonzero and must not compare exact counts across
backends.

## Command and synchronization boundary

`CommandEncoder` supplies copies, uploads, transitions and pass commands. Deferred
backends record native commands; GL executes each command as the encoder method is
called. Raster passes own their attachment/load-store contract; a pipeline, bind
group, vertex/index buffers, viewport/scissor and draw are lowered by the selected
backend. The extras interface holds optional raster operations such as indirect
and multi-draw, with independent capability facts.

## GL immediate execution

GL 4.x, GLES 3.x and WebGL2 run on their context owner thread in a single-threaded
async runtime. The GL backend implements `ImmediateBackend` and does not implement
`DeferredBackend`; the deferred `Queue<Gl>::submit` path is therefore unavailable
at the type level. DX12, Vulkan and WebGPU use the deferred path and do not expose
`ImmediateBackend` merely to share GL's execution model.
This is a static trait distinction, not a runtime mode enum: those backends
have no `Immediate` choice or fallback.

Each GL encoder operation validates and issues its GL/WebGL calls before returning.
Closing an encoder or finishing an `ImmediateSession` reports the work already
issued; it does not submit, replay, reorder or batch a Rust command list. The GL
backend has no GPU queue submission operation. Presentation still performs the
platform's swap/present operation, and a requested completion point may require
a native fence and flush so that GPU completion can be observed. These are
separate from command submission and must not turn into an internal scheduler.

Context and resource owners are thread-affine. `Rc` and local borrowing can enforce
that ownership; the backend does not add a cross-thread command worker, blocking
mutex around GL calls, CPU command cache or deferred replay. Its state cache may
remember known native bindings and invalidate them after uncertainty, but it may
not store commands for later execution. Async completion waits yield to the
single-threaded runtime while the GPU is unfinished; polling must not block that
thread waiting for a fence.

## Browser WebGL2 context lifecycle

The Host owns canvas creation, DOM event registration, CSS sizing, and restoration
policy. One `GlInstance` represents one WebGL context generation. The Host listens
for `webglcontextlost`; when restoration is desired it calls `preventDefault()`
and forwards the event to `GlInstance::notify_context_lost()`. This makes the
old device terminal, wakes pending completion observers with `DeviceLost`, and
makes its surface leases stale. The Host must not reuse resources from that
generation.

After `webglcontextrestored`, the Host creates a new `GlInstance`, Core
`Instance`, device, and surface for the same canvas. Resize remains a Host
lifecycle event: update the canvas backing size and reconfigure the RHI surface
before acquiring another image. An out-of-band backing-size change makes
`acquire()` return `SurfaceOutdated` until that reconfiguration succeeds.

The caller provides ordered work and transitions. RHI does not construct a render graph,
perform global hazard analysis, or decide resource retirement. Submission and
presentation are separate public operations; a present consumes the acquired surface
image associated with the submitted work.

Generic buffer and texture copies consume resources already transitioned to
`CopySrc` and `CopyDst`; recording a copy leaves those states unchanged. The
caller supplies any later transition. Convenience `write_buffer` and
`write_texture` own their temporary upload transition instead. On DX12 their
destination starts and ends in `COMMON`, and callers must not invoke them while
that destination is in a different explicit state. DX12 surface presentation
and pass-local resolve operations retain the transitions required by those
native operations. The backend does not track a global resource-state history.

## Shader and pipeline boundary

`ShaderDesc` declares the artifact format and entry points. The numbered examples
maintain WGSL as their shader source and generate checked-in DXIL and SPIR-V artifacts
for the native DX12 and Vulkan routes. This offline lowering does not claim that DX12
executes WGSL directly: the runtime DX12 route consumes DXIL, while WebGPU consumes
WGSL. Other backends select a supported runtime or generated artifact route explicitly;
generated native binaries are not additional maintained shader sources.

Pipeline and binding descriptors describe the native operation honestly: vertex layouts,
texture view dimensions, sampler and uniform bindings, attachment formats, depth/stencil
state, and raster state are all part of the route. A backend rejects unsupported formats,
view dimensions, layouts or state combinations instead of substituting a no-op.

## Dynamic buffer binding offsets

For a bind group, `set_bind_group` receives one dynamic offset for each layout
entry whose uniform or storage buffer type enables `dynamic_offset`. The slice
is ordered by ascending binding number, independent of visibility or shader
stage. A backend adds each value to the bound buffer range's start offset,
checks the result against the backend's reported buffer-offset alignment, and
ensures the complete bound range remains inside the native allocation. The
bound range size and layout minimum size do not change when an offset is
applied. A missing, extra, misaligned, overflowing, or out-of-allocation offset
returns a structured error before issuing the native binding call.

## Future work

The API contains vocabulary for several operations whose portable coverage is incomplete.
Those declarations are design surface, not a claim that every backend implements them.
Adding a provider requires a real backend crate, facade exposure, descriptor-consistent
support answers, native lowering, and hardware evidence. Expanding DX12 follows the same
rule: expose a capability only after creation, recording, submission and observable
failure behavior are implemented together.
