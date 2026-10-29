# Vulkan Backend Design

`crates/vulkan` is the deferred, explicit API implementation of Fluxel RHI.
It converts a caller-owned, already ordered Core plan into Vulkan command
buffers, submits the resulting native work, and reports completion facts. It
does not build a render graph, derive barriers, schedule application work, or
decide when a renderer may retire a resource.

The crate supports the native Vulkan surface routes available on the target
platform: Win32 on Windows, Android NDK on Android, and Xlib, XCB, or Wayland
when the matching Unix instance extension is available. A caller owns the
window and display handles for the lifetime of a surface and its image leases.

## Shape of the backend

The public entry points are intentionally small: `VulkanInstance` owns loader
and instance creation; `Vulkan` selects this backend for Core; the typed Core
objects expose a `VulkanDevice`, resources, encoders, surfaces, and completion
receipts. Native Vulkan types stay inside this crate.

```text
Renderer / host adapter
        │  Core typed API and ordered SubmitPlan
        ▼
Core validation and backend SPI
        │
        ▼
Vulkan backend
  instance ── adapter/device selection ── capability facts
  resource ── memory / views / mapping / upload-readback pools
  pipeline ── shaders / layouts / descriptors / raster or compute state
  encoder  ── command-pool lease / primary or secondary command buffers
  submit   ── queue and plan gates / fences / timeline semaphores
  surface  ── WSI swapchain / image lease / acquire-present lifecycle
        │
        ▼
Vulkan loader, logical device, queues, and WSI
```

`backend/mod.rs` is only the owner and composition boundary. Its sibling
modules own narrow native concerns: `resource` and `memory` own allocation,
`pipeline` and `shader` build executable state, `encoder` records commands,
`submit` retains issued work and observes fences, `surface` owns WSI, and
`sparse`, `ray`, `mesh`, `external`, and `interop` isolate optional Vulkan
families. This keeps the public RHI model independent of Vulkan handles while
making native destruction order explicit.

## Instance, adapter, and device creation

`VulkanInstance::new` loads the process Vulkan loader, creates a private Vulkan
1.3 instance, and enables only available instance extensions. The instance
retains its loader dispatch table and outlives every logical device. Adapter
enumeration and opening use the Core instance API; device creation chooses a
graphics queue and, when available and requested, a compatible compute queue.

The device records the features, extensions, limits, queue families, timestamp
support, shader capabilities, and entry-point availability that it actually
enabled. Capability reporting and shader admission read those recorded facts,
never mere physical-device potential. Descriptor indexing, timeline semaphores,
dynamic rendering, multiview, subgroup size control, ray tracing, mesh shading,
sparse binding, external handles, and shader numeric features each require an
enabled route.

## Resource and memory ownership

Resources are native, reference-counted objects that retain `DeviceState`.
Their destructors run while that device is still alive and destroy child Vulkan
objects before backing allocation. Texture views retain their texture, and
recorded work retains every buffer, texture, view, pipeline, descriptor group,
query, upload slice, readback slice, and surface attachment it references
until completion has a known disposition.

Allocation is split by purpose rather than using one hidden heap policy:

- regular buffers and images select compatible native memory and retain their
  own allocation;
- default resource pools provide reusable allocations for qualifying resources;
- explicit heaps and placed resources follow their own ownership route;
- upload and readback pools own staging slices and their alignment rules;
- sparse resources preserve sparse binding metadata and use separate binding
  submission; and
- ray resources, device addresses, and external imports use the requirements
  of their native feature families.

Core supplies resource state and synchronization intent. The resource module
does not guess state from usage flags or maintain a second hazard graph;
`transition` lowers the caller's specified layouts, access masks, and queue
ownership transfers. A queue wait establishes ordering, while a barrier is
still needed where the caller requires memory visibility.

## Recording and submission flow

One Core encoder owns a leased transient command pool and a primary Vulkan
command buffer. Calls lower directly into that buffer; `VulkanEncoder` does
not hold an interpreted CPU command stream for later replay. Raster work uses
dynamic rendering, compute and copy commands use the same encoder, and
secondary work has a separate native recording/execution path with compatible
rendering inheritance checked before execution.

The normal lifecycle is:

1. Core validates device, lane, and plan preconditions, then creates a typed
   Vulkan encoder for the selected graphics or compute lane.
2. The encoder validates Vulkan-specific capability, format, binding, and
   current-pass conditions while recording commands. It retains the native
   objects needed by those commands.
3. Finishing an encoder ends its primary command buffer. Explicit external
   semaphore boundaries close one native segment and begin another; they do
   not create a CPU replay stream.
4. Submission keeps one Core plan contiguous with `submit_gate`; each Vulkan
   queue operation is serialized by the narrower `queue_gate` required for
   externally synchronized Vulkan queues.
5. Ordinary work takes the single-queue path. Work containing a compute lane,
   explicit cross-queue dependency, or sparse bind takes the multi-queue path.
   Timeline semaphore values carry Core cross-queue dependencies when enabled.
6. Fence-owned completion retains command pools, resources, semaphores, and
   native dependencies until success, device loss, or an indeterminate result.
   Dropping an observer stops observation; it cannot revoke issued work.

The gates are not a renderer scheduler: they preserve native submission safety
and the caller's established order without reordering graph work or inventing
dependencies.

## Surfaces and presentation

`VulkanSurface` owns the `VkSurfaceKHR`, swapchain loader, current swapchain,
image views, and replacement history. A surface image is an exclusive
`VulkanSurfaceImage` lease, not a portable texture view. The lease may become
a raster attachment through the Core surface vocabulary and can be presented
only through its matching surface lifecycle.

Acquire is asynchronous from the caller's perspective: a worker owns bounded
CPU waits while `Future::poll` only registers a waker. Reconfiguration replaces
the swapchain under the surface lock, increments the generation, and retains
old swapchains plus their wait semaphores until acquire evidence proves they
are no longer in use. An unhanded dropped lease marks the chain abandoned;
the backend does not pretend that it safely released an image. These rules
reject stale images after resize or replacement and keep present semaphores
from being reused prematurely.

## Pipelines, descriptors, and shaders

Pipeline construction is divided into fixed/raster/compute state, layouts,
descriptor-pool allocation, binding variants, specialization, vertex formats,
and optional pipeline-cache persistence. Bind groups lower to Vulkan descriptor
sets; runtime descriptor arrays and variable descriptor counts are admitted
only when the selected device enabled the corresponding indexing route.

The maintained source contract is WGSL, translated through the selected GLSL
or SPIR-V route as appropriate for the operation. Direct SPIR-V is separately
validated and admitted. Shader checks inspect the declared stage, storage
class, operation, extension/capability requirements, and logical-device facts
recorded at creation. Advanced atomics, subgroup controls, barycentrics, early
depth, cooperative matrices, mesh/ray stages, and numeric storage types do not
become available merely because a SPIR-V artifact parses.

Raster lowering uses dynamic rendering and reconciles Vulkan viewport and
front-face conventions with the shared framebuffer contract. It owns render
attachment, resolve, multiview, depth/stencil read-only, and pipeline-signature
checks. This adaptation stays at the backend boundary so examples and renderer
workloads use one cross-backend coordinate contract.

## Capability model and limits

The public route includes graphics and compute, multiple render targets,
indirect drawing and dispatch, queries, secondary recording, clears/resolves,
pipeline caches, mapping, placed memory, and native interop. Descriptor
indexing, subgroup operations and size control, native 8/16-bit shader/storage
types, multiview, ray/mesh, sparse textures, graphics/compute queues,
GPU-count multi-draw, and external resources depend on enabled features,
extensions, formats, limits, and queue support.

The Core BC, ETC2/EAC, and ASTC format families have device-checked public
routes. GPU validation remains format- and device-dependent; the current
Vulkan GPU fixture covers BC1 block copies. ETC2/EAC and ASTC require a
supporting device for GPU validation. Runtime descriptor-array capacity is a
caller-selected operational bound; it is not a claim that every descriptor
array size accepted by a driver is portable.

Two Vulkan routes remain open: fixed acceleration-structure descriptor arrays
and the enhanced mesh batch/count fixture. Their status, evidence, and
reopening conditions are in [`TODO.md`](../../TODO.md). The supported
cooperative-matrix path is deliberately narrow: only selected row-major
layout, stride, pointer provenance, and tuple combinations are admitted; other
layouts and tuples are not implied.

## Deliberate boundaries

The backend does not infer resource state from creation flags, choose aliases,
fabricate resident-memory statistics, replay commands on the CPU, or enable
features omitted during device creation. Vulkan queue and synchronization
mechanics execute the caller's plan; RenderGraph and Renderer remain
responsible for resource-use planning, hazards, ordering, frame policy, and
retirement decisions.
