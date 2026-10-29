# Fluxel RHI Design

This directory describes the goals, ownership, and high-level architecture of
the `fluxel-rhi` crates. It focuses on what each crate provides, why its
boundaries exist, and how the main pieces work together. API details belong in
the source-level API documentation; test procedures belong in
[`../testing.md`](../testing.md), and the executable coverage matrix belongs in
[`../conformance-matrix.md`](../conformance-matrix.md).

## System boundary

RHI executes GPU work that an upper layer has already planned. It owns typed
resource and device contracts, backend lowering, native synchronization,
surface-image leases, and facts about accepted work and completion. It does not
compile a render graph, infer hazards, choose resource aliases, pace frames, or
decide when renderer resources may be retired.

| Owner | Responsibility |
|---|---|
| `fluxel-host` and platform adapters | Window and canvas lifecycle, borrowed native handles, event delivery, resize, foreground/background, and restoration policy |
| `fluxel-rendering` | RenderGraph, resource-use planning, hazards, ordering, frame policy, and retirement decisions |
| `fluxel-assets` | Persistent asset identity, typed handles, caching, and CPU/GPU residency |
| `fluxel-rhi-core` | Shared operation meanings, typed handles, device identity, capability contracts, and backend SPI |
| RHI backend crates | Native resources, lowering or immediate execution, synchronization, surface leases, and native in-flight ownership |
| `fluxel-rhi` facade | One import namespace that reexports Core and feature-selected concrete backends |

The host does not depend on RHI. A host-to-RHI adapter borrows a native window
handle and creates a surface; it does not transfer window ownership. Browser
and Android adapters similarly translate platform lifecycle events without
owning a GPU backend. RenderGraph and Renderer depend on RHI, not vice versa.

## Workspace architecture

```text
Application / Host adapters / Renderer
                    |
             fluxel-rhi facade
              /           \
             v             v
           Core <-------- backends

Backends: DX12, Vulkan, WebGPU, GL family, Metal
```

Core is backend-independent. Each backend implements Core's hidden SPI and
returns concrete `Device<B>` and resource types. The backend type is retained
through normal calls, so GPU hot paths do not need a runtime backend enum,
downcasts, or a global `Send + Sync` requirement. Applications choose a
backend at their integration boundary. Device capabilities remain runtime
facts because adapters and enabled features differ.

The facade in `crates/rhi` reexports the Core API and conditionally exposes
backend modules for supported targets and Cargo features. It does not define a
second resource model or runtime dispatcher. See [`core.md`](core.md) for the
shared contract.

## Execution models

DX12, Vulkan, WebGPU, and Metal use deferred native recording: RHI operations
encode into native command objects and submit them through the selected
backend. Deferred recording is not a portable CPU command list and does not
imply parallel recording.

The GL family uses immediate execution because its operations are bound to the
current context thread. Calls issue native GL work as they are encoded; the
backend does not record commands for later replay. Core exposes these models
through distinct traits rather than making one backend pretend to support the
other model.

Both models preserve the same Core meanings for ordered work, failures,
presentation, and completion. A submitted or issued prefix can outlive a
failure. Backends retain native dependencies until completion or a terminal
failure establishes their disposition. Completion evidence is not a renderer
retirement decision.

## Crate design pages

- [Core contract and trust boundary](core.md)
- [Direct3D 12 backend](dx12.md)
- [Vulkan backend](vulkan.md)
- [WebGPU backend](webgpu.md)
- [GL, GLES, and WebGL2 backend](gl.md)
- [Metal backend](metal.md)
- [Open design TODOs and deferred work](todo.md)

## Shared design rules

- Core resource handles are associated with a device identity. Cross-device
  misuse is rejected before reaching a driver; native pointers, browser
  sessions, context handles, and Objective-C objects remain backend-private.
- The caller supplies an already ordered execution plan. RHI validates device,
  lane, and predecessor facts but does not sort a graph or infer resource
  hazards.
- Resource transitions and cross-queue waits remain explicit. A queue wait
  does not invent memory visibility barriers.
- Backend raster and copy adapters reconcile native viewport, winding, and
  transfer conventions with the shared scene and texture contract. Examples
  keep one maintained scene and shader source across backends; backend
  differences are corrected at the backend boundary, not by changing the
  shared workload.
- Capability reports describe an implemented native route on the selected
  device. An upstream enum, physical-device potential, shader format, or visual
  fallback alone does not establish support.
- Errors preserve whether native work was rejected, partly issued, accepted, or
  left uncertain. Returning from encoding or presenting is not proof of GPU
  completion.
- Surface images are generation-bound exclusive leases. The host owns the
  window or canvas lifecycle; the backend owns the GPU presentation path.

## Source and status authority

The `crates/core/src/api/` modules define the public contract, and
`crates/core/src/backend/` defines the backend SPI. Per-backend implementation
and device capability code determine which parts of that contract are
available. Current unfinished work and its reopening conditions are tracked in
[`todo.md`](todo.md). Historical design notes and review logs are not normative.
