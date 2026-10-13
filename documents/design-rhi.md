# Fluxel RHI execution contract

**Status:** current implementation and design boundary.

## Public API and ownership

`fluxel_rhi_core::api` is the single Core API namespace. The facade re-exports it as
`fluxel_rhi::api`; applications use that namespace and select a concrete backend at
initialization. On Windows the facade exposes `fluxel_rhi::dx12::Dx12` and
`fluxel_rhi::vulkan::Vulkan`; browser WASM exposes `fluxel_rhi::webgpu::WebGpu`.
Each route has an optional feature.

The API is statically dispatched: `Device<B>`, resources, pipelines, encoders and
recorded work retain their concrete `Backend` type. RenderGraph or its host owns graph
scheduling, hazard analysis, lifetime policy and frame pacing. The RHI creates native
objects, encodes the supplied operations, submits them, and reports support or failure.

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

GL and Metal are not active workspace backends. Documentation and CI
must not treat source paths or compile checks as hardware execution evidence.

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

## Command and synchronization boundary

`CommandEncoder` records copies, uploads, transitions and pass commands. Raster passes
own their attachment/load-store contract; a pipeline, bind group, vertex/index buffers,
viewport/scissor and draw are lowered by the selected backend. The extras interface
holds optional raster operations such as indirect and multi-draw, with independent
capability facts.

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

`ShaderDesc` declares the artifact format and entry points. DX12 currently accepts DXIL
on its direct route. Examples that need DX12 use checked-in HLSL source and DXIL assets;
WGSL or GLSL must not be described as directly executable on DX12 unless a real
translation route is implemented and selected.

Pipeline and binding descriptors describe the native operation honestly: vertex layouts,
texture view dimensions, sampler and uniform bindings, attachment formats, depth/stencil
state, and raster state are all part of the route. A backend rejects unsupported formats,
view dimensions, layouts or state combinations instead of substituting a no-op.

## Future work

The API contains vocabulary for several operations whose portable coverage is incomplete.
Those declarations are design surface, not a claim that every backend implements them.
Adding a provider requires a real backend crate, facade exposure, descriptor-consistent
support answers, native lowering, and hardware evidence. Expanding DX12 follows the same
rule: expose a capability only after creation, recording, submission and observable
failure behavior are implemented together.
