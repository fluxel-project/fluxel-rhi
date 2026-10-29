# Direct3D 12 Backend Design

`fluxel-rhi-dx12` is the Windows-native implementation of the shared RHI
contract. It translates typed Core operations into D3D12 command lists, DXGI
presentation, and, where admitted, DXR operations. The application still owns
render-graph planning, resource-use ordering, frame pacing, and retirement
policy. This crate owns the native objects that execute that already planned
work and reports facts about its completion.

The shared contract is designed first for RenderGraph and rendering-pipeline
implementations. This backend keeps execution thin and low-cost by preserving
direct D3D12 capabilities and exposing their execution facts to graph planning;
application-level developer convenience is secondary.

The public entry points are `Dx12Instance`, `Dx12`, and the typed
`Device<Dx12>`. Native D3D12 and DXGI types remain private to this crate.
Callers work through the Core API, so a DX12 object cannot be confused with an
object from a different backend or device.

## What makes this backend different

D3D12 is a deferred, explicit API. An encoder creates a D3D12 command list;
recording validates the Core operation, emits D3D12 calls, and gathers every
native object that must outlive execution. Submission executes that list on a
native queue and returns a fence-backed completion point. Therefore command
recording is separate from execution, and completion is an observed GPU fact,
not an inference from a successful Rust call.

The backend supports a direct graphics queue and opens a separate compute
queue when the requested device exposes that lane. Queue ordering and waits
are explicit parts of the Core submission plan. The backend serializes native
queue access and fence allocation, but it does not reorder work, create
dependencies, or turn the queues into a render scheduler.

DXGI presentation is also explicit. A surface borrows a live Win32 `HWND`,
owns a flip-model swapchain generation and its RTVs, and issues `Present` only
when the acquired image lease is handed off. Host window lifetime and window
policy remain outside the crate.

## Layers

The implementation is organized by responsibility rather than by a second
public abstraction layer:

```text
Core typed API and backend SPI
        |
Dx12Instance / Dx12Device
        |
  +-----+-------------------+---------------------+
  |                         |                     |
resource, memory,       pipelines, shaders,    surface and
descriptor modules      raster, compute, ray    DXGI image leases
  |                         |                     |
encoder records D3D12 command lists and retained native owners
        |
submission preflight, direct/compute queues, D3D12 fences
        |
completion polling or asynchronous fence wait
```

`backend/mod.rs` creates the device-wide state: `ID3D12Device`, queues,
fences, descriptor heaps, command allocator pools, upload pages, default
resource pools, pipeline helper caches, and the in-flight owner lists.
Focused modules own resource descriptions and views, descriptor layouts and
writes, pipeline construction, shader compilation, queue mechanics, surface
leases, and feature families such as sparse, mesh, or ray work. The encoder is
the only layer that turns Core command operations into native list calls.

The Core SPI preserves a concrete `Dx12` type for devices, buffers, textures,
views, bind groups, encoders, surfaces, and completion points. The public
`Device<Dx12>` identity is checked before a native object is used; the crate
does not expose COM pointers as general RHI resource handles.

## Device opening and capability facts

`Dx12Instance::new` opens a DXGI factory. Adapter enumeration happens through
the Core instance API, and `open` creates the D3D12 device only for the chosen
adapter. Opening records fixed device facts such as feature level, shader
model, resource-heap tier, enhanced-barrier support, mesh and ray support,
sparse support, descriptor-indexing prerequisites, and view-instancing tier.

Those facts become `DeviceCapabilities` and `DeviceLimits`. A feature is
reported only when both the native device and this backend's exact lowering
route can support it. For example, a D3D12 feature tier alone does not imply
that every shader format, descriptor shape, texture format, or stage is
accepted.

The optional debug callback is a diagnostic aid. It does not make driver
validation output a substitute for completion or visual correctness.

## Resource and descriptor ownership

Buffers and textures use D3D12 resources with typed Core wrappers. Ordinary
default-heap allocations come from backend-owned pages when native alignment
and heap rules allow it; their allocation lease stays alive with the resource.
Committed resources and explicitly placed resources retain their independent
native backing. The default allocator neither aliases resources on behalf of a
caller nor decides when an aliasing barrier is safe.

Descriptor layout, descriptor writing, and bind-group ownership are separate
from command recording. Device-owned shader-visible heaps provide GPU tables;
the CPU descriptor pool owns non-shader-visible views. Pipeline layout creation
maps the validated Core binding layout to D3D12 root parameters and descriptor
tables. Raster and compute encoders bind those tables only after checking the
pipeline, group, and dynamic-offset shapes agree.

Upload and readback have dedicated paths. CPU writes are copied from upload
storage into default resources; texture readback targets a compatible readback
buffer and requires the declared copy state. These helpers insert only their
own required native transitions. They do not infer transitions or hazards for
ordinary caller-owned resources.

## Shader and pipeline routes

Shader route selection is explicit and precedes pipeline creation:

- Direct DXIL is accepted as a D3D12 artifact after entry-point validation.
- HLSL is a direct source route when the selected shader model and stage are
  admitted; DXC compilation occurs for the pipeline specialization that needs
  it.
- WGSL follows the maintained Naga-to-HLSL/DXC route. The compiler validates
  entries, bindings, stage declarations, workgroup size, overrides, and
  required feature declarations against the pipeline layout.

The result is a `Dx12Shader` plus entry-specific native bytecode used to build
raster, compute, mesh, or ray pipeline objects. Pipeline caching applies to
the compatible fixed pipeline forms it owns. DXR state objects are separate
and are intentionally outside that cache.

Direct DXIL, direct HLSL, and WGSL-lowered DXIL are different artifact routes.
Success on one says nothing about the others. The same applies to optional
shader capabilities: admission combines the device fact, required Core
feature, accepted stage/storage class, and translator support.

## Recording, submission, and completion

The normal run path is:

```text
create resources and pipelines
        -> begin encoder
        -> record ordered Core commands into a D3D12 list
        -> finish into retained recorded work
        -> preflight plan, execute lists, signal queue fence
        -> receive checkpoint/completion point
        -> poll or await its native fence
        -> retire retained owners after that fence completes
```

Recording applies explicit Core transitions, clears, copies, render passes,
bind groups, draw/dispatch operations, queries, indirect commands, and
secondary bundles to the command list. Secondary work is a D3D12 bundle with
its own allocator; the direct encoder retains it after `ExecuteBundle` so the
allocator, list, pipelines, descriptors, and resources survive execution.

Submission preflight validates the packet before native handoff. It separates
direct and compute segments, issues caller-specified native queue waits, and
signals the appropriate queue fence. A checkpoint covers only the ordered
prefix on that lane. A direct-queue fence does not prove compute completion
unless the submitted plan explicitly joined the queues.

Recorded work retains command allocators, upload resources, descriptor owners,
pipeline and resource references, query bookkeeping, and other native
dependencies in an in-flight list keyed by its fence value. `completion_status`
reads `ID3D12Fence::GetCompletedValue`; only a completed native fence releases
those owners. CPU waiting uses an event registered with that fence instead of
busy waiting. A value of `u64::MAX` represents device removal or indeterminate
accepted work, so it is quarantined rather than treated as a canceled packet.

## Surface lifecycle

`Dx12::surface_from_raw_window_handle` accepts only a Win32 handle and carries
the safety requirement that the host keeps it alive while the surface and all
image leases exist. Configuration creates or replaces a DXGI flip-model
swapchain and increments the surface generation. The surface permits one
exclusive acquired image at a time; an unpresented lease clears acquisition on
drop. Resize or reconfiguration makes old-generation images stale, and a
handoff validates generation before calling `Present`.

The surface supports its declared formats, FIFO presentation, and immediate
presentation only where DXGI reports tearing support. It owns swapchain state,
not an application frame loop, fullscreen policy, or host event processing.

## Capability boundaries

The implemented public route includes graphics and compute, multiple render
targets, indirect drawing and dispatch, queries, secondary recording,
standalone clears and resolves, placed memory, scoped native/external interop,
and pipeline caches. Raster supports up to eight color attachments,
per-target blend and write masks, depth/stencil and read-only depth/stencil
views, attachment clears, and resolves. Formats, sample counts, shader model,
resource descriptors, and device tiers remain part of admission.

Descriptor arrays and indexing are constrained by device capabilities,
descriptor-table layout, and shader route. The verified WGSL non-uniform
uniform-buffer and storage-buffer routes are fixed, fully bound, compute-only
shapes. They do not promise runtime or partially bound arrays, dynamic offsets,
other stages, or mismatched shader/layout metadata.

BC1–BC7 compressed textures use DXGI mappings with checked creation,
block-aligned upload/copy, and sampling. `texture_support` remains the source
for a particular format, dimension, and usage. ETC2/EAC and ASTC have no DXGI
mapping in this backend.

Ray tracing, ray query, mesh shaders, sparse resources, dynamic raster state,
texture atomics, GPU-counted draws, multiview, and source wave-size selection
are individually gated. They are available only when the selected adapter and
the exact descriptor, pipeline, and shader path qualify.

## Non-goals

This crate does not implement RenderGraph scheduling, global resource-state
tracking, CPU command replay, generic device-loss notification policy,
arbitrary shader format conversion, or a DXR state-object cache. Open
cross-backend work and its reopening conditions are recorded in
[`TODO.md`](../../TODO.md).
