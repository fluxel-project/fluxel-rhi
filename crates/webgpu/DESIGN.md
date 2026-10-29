# WebGPU Backend Design

`crates/webgpu` implements the Core contract through browser WebGPU on
`wasm32`. Browser objects and JavaScript ownership stay inside the backend and
its owner thread. Core receives typed resources and execution facts, not DOM,
`GPUDevice`, canvas, or JavaScript session objects.

The shared contract is designed first for RenderGraph and rendering-pipeline
implementations. This backend preserves native WebGPU execution where possible.
Its current Core capabilities report whether an operation is available, but do
not yet describe every cost-bearing lowering. A route that requires GPU
preprocessing, temporary resources, CPU iteration, or an internal pass split
must be made visible to graph planning; this gap is tracked in the root
[design TODO](../../TODO.md#p7-lowering-profile).

The backend has one browser-selected adapter and one ordered queue per opened
device. It does not turn browser WebGPU into a desktop-style multi-queue API,
a portable command recording format, or an owner of DOM and application
lifecycle.

## Backend character

- `WebGpuDevice` is `Rc`-backed and owner-thread local. Browser objects stay on
  the thread where JavaScript may use them; a global lock does not make a
  browser device transferable.
- Browser adapter selection is asynchronous. Core exposes one default-adapter
  ordinal rather than inventing a physical-adapter list that browsers do not
  provide.
- The browser queue maps to one graphics lane. Same-queue order is supported;
  independent queues, queue-family ownership and sparse binding are not
  fabricated.
- JavaScript promises become Core futures. Dropping a Rust future cancels only
  observation: native submitted work and its browser roots remain alive through
  a terminal outcome.
- A canvas current texture is an exclusive, generation-bound lease. It is
  separate from durable application textures.

## Layered architecture

```text
Core typed API and backend SPI
            |
 instance / resource / pipeline / command / surface adapters
            |
 WebGPU ownership, validation, and in-flight retention
            |
 wasm-bindgen JavaScript bridge and browser promises
            |
 GPUDevice, GPUQueue, GPUCanvasContext, GPUBuffer, GPUTexture
```

`instance.rs` owns browser adapter/device opening, capability discovery,
validation, and device-loss listeners. `resource.rs`, `bindings.rs`, and
`resource_validation.rs` translate Core descriptors only after format, usage,
limit, and binding checks. `pipeline.rs` and shader modules select a shader
route and make pipelines. `command/`, `raster.rs`, `compute.rs`, and texture
operation modules encode native WebGPU calls. `submit/` owns queue submission,
completion promises, retention, and retirement.

Surface images live in `surface.rs`, separate from normal resources because a
canvas current texture is borrowed per frame. Mapping/readback, query
segments, unaligned transfers, clear/blit/resolve paths, and GPU-counted
indirect handling have dedicated modules: their browser restrictions must be
lowered explicitly and never silently through CPU readback.

## Device opening and capability facts

Opening requests the browser default adapter, then requests only features and
limits that adapter reports. Compression formats, timestamps, `shader-f16`,
subgroups, depth-clip control, dual-source blending, and indirect
first-instance are device facts, not claims about every browser. The resulting
Core capabilities and limits describe that opened device.

Device state retains its `GPUDevice`, queue, monotonic submission sequence,
in-flight records, upload-page arena, uncertain native roots, per-device
utility pipelines, and validation-listener closure. It installs both an
uncaptured validation-error listener and the `GPUDevice.lost` promise. Either
event fails observable completion points, but does not reclassify issued work
as unissued or prematurely discard objects the browser may still access.

## Browser ownership and command flow

The device and browser resources are owner-thread local. RHI commands encode
directly into `GPUCommandEncoder` and native pass encoders; they are not stored
as a portable CPU command stream. Submission finishes the native command
buffer and sends it to the WebGPU queue.

```text
request adapter/device -> report capabilities -> create typed resources
        -> encode GPUCommandEncoder/pass calls -> finish -> GPUQueue.submit
        -> queue completion promise -> retire retained roots -> Core completion
```

At submission the backend rejects unsupported plan shapes, including sparse
binds, and processes accepted steps in the browser queue's order. Requested
step completions become queue boundaries for those plan steps; end-of-submit
completion retains the final boundary. Completion proves progress of native
work; Renderer and RenderGraph decide when logical resources can be retired.

The encoder retains JavaScript roots required by the submitted work. Queue
completion settles a submitted prefix and permits those roots to be released.
Dropping a Rust completion observer cancels observation, not native work. On
device loss, completion fails while the backend preserves native dependencies
until the browser establishes their terminal disposition.

In-flight records retain command buffers, bindable objects, surface handoffs,
upload pages, and other JavaScript roots needed beyond the Rust call stack.
Retirement releases only entries proven to precede a settled queue boundary and
returns upload pages to the per-device pool. If a queue boundary becomes
uncertain, retained values stay associated with a monotonic sequence until a
later confirmed browser boundary makes release safe.

## Resources and data movement

Typed wrappers retain `GPUBuffer`, `GPUTexture`, `GPUTextureView`,
`GPUSampler`, layouts, and bind groups behind Core handles. Core validates
device identity before reaching the backend; descriptor lowering also checks
native format, usage, view, and binding rules. A raw JavaScript value never
becomes public resource identity or a cross-device escape hatch.

Upload pages stay allocated until the submission that uses them completes.
Unaligned buffer or texture shapes use retained staging or a GPU-side utility
operation when WebGPU alignment/copy rules require it. Readback maps dedicated
buffers asynchronously. Clear, blit, resolve, depth/stencil conversion, query
packing, and indirect-count packing use per-device utility pipelines where
WebGPU lacks the exact Core operation shape; none of these paths uses hidden
CPU readback to implement GPU work.

## Surface and shader architecture

The backend configures the WebGPU canvas and manages the generation-bound
current-texture lease. DOM sizing, canvas lifecycle, frame scheduling, and
device restoration belong to the host adapter. A resize or reconfiguration
invalidates stale surface images.

WGSL is the native shader language. SPIR-V translation and validation run in a
browser worker that owns no WebGPU objects; GPU resources and command encoding
remain on the browser owner thread. Upload alignment and unsupported copy
shapes use retained staging pages or GPU-side repacking rather than hidden CPU
readback.

WGSL is compiled directly by `GPUDevice.createShaderModule`; its target records
the browser-WebGPU group/binding ABI. SPIR-V is accepted only after the
application configures the dedicated translation worker before shader creation.
That worker translates and validates SPIR-V to WGSL but owns no `GPUDevice`,
resource, surface, or encoder. The browser owner thread creates the final
shader module, so this is explicitly `SPIR-V -> worker -> WGSL -> WebGPU`, not
a claim that WebGPU executes Vulkan SPIR-V directly.

Pipeline creation uses the browser asynchronous pipeline API. Immediate/push
constant forms, subgroups, and optional raster features are admitted against
WebGPU language features and opened-device limits. Dynamic depth bias,
non-fill polygon modes, conservative rasterization, and arbitrary multiview
are rejected when standard WebGPU has no corresponding mechanism.

## Surface lifecycle

`WebGpu::surface_from_canvas` borrows a host-owned `HtmlCanvasElement` and its
`GPUCanvasContext`; it creates no window and owns no browser frame loop. A
surface is bound to one device and has a configuration generation.

1. The host changes canvas dimensions and then Core configures the surface.
   The backend checks format, usage, alpha/color space, device limits, and the
   current canvas dimensions before calling `GPUCanvasContext.configure`.
2. Acquisition calls `getCurrentTexture`, makes its one renderable view, and
   returns a lease stamped with that generation. A second live lease is
   rejected.
3. Submission can hand off only a lease with the current generation; a resize
   or reconfiguration makes an old lease `SurfaceOutdated`.
4. Dropping an unhanded lease marks the canvas abandoned. A later configuration
   first unconfigures the browser context, avoiding reuse of a discarded
   current texture.

Canvas presentation belongs to the browser's canvas lifecycle rather than a
separate native swapchain call. Resize, page lifecycle, device replacement,
and restoration policy are host-adapter responsibilities. Restoration opens a
new device generation; it does not revive old resource handles or leases.

## Capabilities and video boundary

Optional standard features are requested and reported only when exposed by the
adapter. Format, usage, binding, and operation checks remain specific to the
native route. GPU-counted indirect execution transforms arguments on the GPU;
it does not read the count on the CPU or replay application commands.

`VideoFrame` import is a separate external-video binding domain. It wraps an
opaque source and keeps the JavaScript source root alive through queue
completion. The backend does not decode video, expose plane formats, convert
color, schedule frames, or own source creation and closure. Other backends do
not claim this route. Real browser-source integration remains open under
[`TODO.md`](../../TODO.md).

## Public feature coverage

The public route includes graphics and compute, MRT, depth/stencil, dynamic
offsets, indirect draw/dispatch, multi-draw including GPU-count lowering,
standalone clears, copies/blits/resolves, buffer mapping/readback, occlusion,
and optional timestamp queries.

Adapter features and limits gate compressed formats, immediates, subgroups,
dual-source blending, and unclipped depth. Ray/mesh, sparse resources, native
multiple queues, multiview, complete pipeline statistics, native heap
placement, and serialized pipeline caches have no matching standard browser
route.

Compressed formats are optional adapter features. The current AMD browser
evidence covers BC1; ETC2/EAC and ASTC GPU validation remains pending. ASTC
HDR has no mapping in this route.

Standard WebGPU does not provide routes for every Core domain. In particular,
ray pipelines, mesh shaders, sparse binding, independent native queues,
placement heaps, native secondary recording, full pipeline statistics, and
some presentation/copy shapes remain unavailable unless a concrete WebGPU
mechanism and validation support them. Unsupported operations fail explicitly.
