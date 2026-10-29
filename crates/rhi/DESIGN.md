# Fluxel RHI Design

This directory describes the goals, ownership, and high-level architecture of
the `fluxel-rhi` crates. It focuses on what each crate provides, why its
boundaries exist, and how the main pieces work together. API details belong in
the source-level API documentation. Test procedures, hardware evidence rules,
and backend coverage matrices belong in the
[verification guide](../../documents/verification.md).

## Design intent

Fluxel RHI is designed first for RenderGraph and rendering-pipeline
implementations. It provides a thin, low-cost execution layer that unifies GPU
operation semantics while preserving backend-native capabilities and exposing
execution facts that affect graph planning. It deliberately prioritizes
predictable execution of already planned work over an application-level
convenience API.

## System boundary

RHI executes GPU work that an upper layer has already planned. It owns typed
resource and device contracts, backend lowering, native synchronization,
surface-image leases, and facts about accepted work and completion. It does not
compile a render graph, infer hazards, choose resource aliases, pace frames, or
decide when renderer resources may be retired.

- **`fluxel-host` and platform adapters:** window/canvas lifecycle, borrowed
  native handles, event delivery, resize, foreground/background, restoration.
- **`fluxel-rendering`:** RenderGraph, resource-use planning, hazards, order,
  frame policy, and renderer retirement decisions.
- **`fluxel-assets`:** persistent asset identity, typed handles, caching, CPU
  and GPU residency.
- **`fluxel-rhi-core`:** shared operation meanings, typed handles, device
  identity, capability contracts, and backend SPI.
- **RHI backends:** native resources, lowering or immediate execution,
  synchronization, surface leases, and native in-flight ownership.
- **`fluxel-rhi` facade:** one import namespace for Core and selected backends.
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
second resource model or runtime dispatcher. See the [Core design](../core/DESIGN.md)
for the shared contract.

## Runtime path and ownership flow

A typical frame crosses the workspace in this order:

```text
Host event / frame policy
    -> Renderer or caller builds ordered work
    -> Core Instance selects adapter and Device<B>
    -> backend creates typed resources and Surface
    -> encoder records native work (or GL issues it immediately)
    -> Queue submission reports acceptance and completion token
    -> Surface presents an acquired generation-bound image
    -> Renderer observes completion before retiring its resources
```

1. **Platform setup.** `fluxel-host` owns windows, event pumps, and native
   handles. A narrow adapter borrows the platform handle to create a backend
   surface or context; the RHI does not own the window. Browser adapters provide
   canvas and lifecycle events through their platform route.
2. **Backend selection.** The application selects a concrete backend at its
   integration boundary. The facade exposes feature-gated modules, while Core
   supplies the common API. Backend types stay static in `Device<B>` and
   related objects; there is no hidden global runtime dispatcher.
3. **Capability-driven construction.** The selected adapter reports device and
   operation capabilities from implemented native routes. Resource, pipeline,
   and surface creation validates those facts and associates objects with a
   device identity and generation.
4. **Ordered execution.** Renderer or caller supplies dependency order,
   resource transitions, and cross-queue waits. Core validates the contract;
   backend code translates it into API calls and preserves failures at the
   point they occur. RHI does not infer graph hazards or add missing barriers.
5. **Presentation and completion.** Surface acquisition returns an exclusive
   image lease tied to a surface generation. Queue acceptance, GPU completion,
   and successful presentation are separate facts. Backend-owned in-flight
   references keep native objects alive until completion or terminal failure.
6. **Retirement.** Completion evidence returns to Renderer, which decides when
   its logical resources can be retired. RHI reports what the device finished;
   it does not decide resource aliasing, frame pacing, or renderer retirement.

## Crate responsibilities

- **`fluxel-rhi-core`:** backend-independent API contracts, typed objects,
  capability semantics, structured errors, and hidden SPI; it does not choose
  a GPU API.
- **`fluxel-rhi-dx12`:** D3D12 device, queues, resources, command recording,
  presentation, and completion; Windows only.
- **`fluxel-rhi-vulkan`:** Vulkan instance/device selection, explicit resources
  and synchronization, command submission, surfaces, and completion.
- **`fluxel-rhi-webgpu`:** WebGPU device/resource/command mapping plus browser
  surface and lifecycle integration.
- **`fluxel-rhi-gl`:** desktop GL, GLES, and WebGL2 immediate context-thread
  execution with profile-specific admission.
- **`fluxel-rhi-metal`:** Apple Metal device/resource/command mapping,
  presentation, and completion; Apple targets only.
- **`fluxel-rhi`:** feature-selected facade and examples; it reexports Core and
  selected backends, rather than implementing another backend layer.
Each crate's `DESIGN.md` follows its responsibilities from public adapter down
to native execution and explains backend-specific limits. Package `README.md`
files explain the crate's role in an RG-oriented renderer or platform adapter.
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

## RenderGraph-to-RHI execution contract

RenderGraph compiles resource use, ordering, transitions, queue dependencies,
and lifetimes before execution. RHI validates and executes that contract; it
does not infer missing hazards, choose aliases, or repair an incomplete plan.
This keeps graph decisions in the graph while allowing RHI to preserve native
queues, barriers, resource binding, and completion mechanisms.

For each operation, the graph needs four kinds of facts:

- **Input preconditions:** required usage, ownership, resource state, lane, and
  lifetime facts that must already be established by the compiled plan.
- **Execution route:** whether the backend issues a native operation, several
  native operations, or a backend lowering, and whether the route is supported
  by this opened device.
- **Extra work:** any hidden GPU preprocessing, temporary allocation, internal
  synchronization, CPU iteration, or pass split that can affect graph cost or
  attachment correctness. Such differences must be queryable before execution
  when they affect graph planning; they must not be disguised as native support.
- **Completion evidence:** whether work was rejected, partially accepted, or
  left indeterminate, and which completion point proves native progress. The
  renderer uses this evidence to decide resource retirement.

The current capability model reports support, but it does not yet expose a
general lowering profile for cost-bearing routes such as WebGPU GPU-count
draws. See the open design TODO for that gap and its acceptance conditions.

## Crate design pages

- [Core contract and trust boundary](../core/DESIGN.md)
- [Direct3D 12 backend](../dx12/DESIGN.md)
- [Vulkan backend](../vulkan/DESIGN.md)
- [WebGPU backend](../webgpu/DESIGN.md)
- [GL, GLES, and WebGL2 backend](../gl/DESIGN.md)
- [Metal backend](../metal/DESIGN.md)
- [Open design TODOs and deferred work](../../TODO.md)

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
[`TODO.md`](../../TODO.md). Historical design notes and review logs are not normative.
