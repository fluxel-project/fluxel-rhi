# Fluxel RHI execution contract

**Status:** current implementation and design boundary.

## Public API and ownership

`fluxel_rhi_core::api` is the single Core API namespace. The facade re-exports it as
`fluxel_rhi::api`; applications use that namespace and select a concrete backend at
initialization. On Windows the facade exposes `fluxel_rhi::dx12::Dx12`.

The API is statically dispatched: `Device<B>`, resources, pipelines, encoders and
recorded work retain their concrete `Backend` type. RenderGraph or its host owns graph
scheduling, hazard analysis, lifetime policy and frame pacing. The RHI creates native
objects, encodes the supplied operations, submits them, and reports support or failure.

A backend must keep public descriptors, support queries and command results aligned. A
feature is advertised only if the corresponding native route exists. Format- and
descriptor-specific limits belong in support queries or operation results. `Unsupported`
is the required result for a route that is absent.

## Current repository shape

- `crates/core/src/api/` defines the public vocabulary and backend SPI traits.
- `crates/rhi` is the typed facade and contains the numbered runnable examples.
- `crates/dx12` is the active native backend. Its lowering is organized under
  `backend/`, including `encoder/`, `pipeline`, `texture`, `descriptor`, `sampler`,
  `surface`, `query`, and `readback`.

There are no GL, Vulkan, Metal, or WebGPU backend crates in this checkout. Documentation
and CI must not treat paths for those backends as implementation or hardware evidence.

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
