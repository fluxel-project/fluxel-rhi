# GL-Family Backend Design

`crates/gl` is one statically dispatched backend for Desktop OpenGL through
WGL, GLES through EGL, and browser WebGL2. Shared lowering keeps operation
meanings consistent while separate providers and capability discovery preserve
the different limits of each profile.

## Immediate owner-thread execution

GL calls execute on the thread where the selected context is current. Encoding
issues native calls immediately; the backend does not build a deferred command
stream or replay application work on another thread. Rust ownership and
thread-affine objects express this constraint directly instead of hiding it
behind a global mutex or worker queue.

The backend uses a context identity plus slot/generation information so reused
numeric GL object names cannot make stale resources appear valid. Device and
resource identities are checked before native access.

## Layering and state knowledge

The shared GL family separates typed operation lowering, Core SPI adaptation,
native WGL/EGL providers, browser WebGL2 ownership, and context state
management. Desired state and applied state are tracked separately. The cache
is an optimization, not a resource-state or hazard graph; uncertainty, raw
interop, object deletion, or context loss invalidates affected knowledge.

Each capability is derived from the selected profile's version, extensions,
limits, and callable entry points. Desktop-only support does not leak into
GLES or WebGL2. WGSL lowers to the admitted GLSL profile with reflected
bindings and coordinate adaptation. Desktop GL SPIR-V is a separate artifact
ABI and is not treated as Vulkan SPIR-V execution.

## Queries, mapping, and surfaces

Completion observation is non-blocking and owner-thread-affine. Native fences,
query publication, GPU staging copies, and CPU result reads are separate
events. The backend does not use `glFinish`, per-draw error draining, or hidden
CPU readback as completion shortcuts.

Desktop and GLES surface providers borrow their host drawable. WebGL2 uses a
canvas surface and browser-owned context. Surface generations reject stale
images after resize or context loss. Context restoration creates a new backend
generation; it does not resurrect old resources.

Scoped native interop is available on native WGL/EGL contexts under a
thread-affine guard. The guard borrows the dispatch table, restores required
state on exit or unwind, and invalidates cached state. It does not export RHI
object names or provide a raw deferred command recorder. WebGL2 has no native
dispatch-table pointer.

## Profile-specific constraints

- **Desktop GL4+/WGL:** desktop versions and extensions may expose query-buffer,
  SPIR-V, multiview, and other routes after exact entry-point checks. Multiview
  is limited to supported contiguous ranges; it does not imply arbitrary view
  masks.
- **GLES/EGL:** limits and extensions are queried per context. Persistent
  mapping, storage-buffer indexing, depth/stencil transfers, and GPU-counted
  draws require their own extension or version path.
- **WebGL2:** runtime arrays are specialized to bounded native slots; mapping
  uses fence-backed staging; texture readback is admitted per format. Context
  loss is terminal for the current generation and restoration is owned by the
  host.

The same-sample multisample texture copy is explicitly unsupported on GLES,
even where an API extension exists, until a complete attachment/readback
route is proven. This does not remove supported desktop routes, single-sample
copies, or standalone resolves.

## Explicitly unavailable domains

The GL family does not implement secondary recording, independent GPU queues,
sparse resources, placed memory, external memory/resources, mesh shaders, ray
pipelines, acceleration structures, or ray query. These are support boundaries,
not commands to emulate the missing feature. WebGL2 additionally lacks
standard routes for GPU-counted multi-draw, full pipeline statistics, native
serialized program binaries, arbitrary multiview masks, compressed color
clears, and several advanced raster modes.

The GLES framebuffer-fetch and Vulkan local-attachment-read family is deferred
until a RenderGraph consumer defines tile-local access and ordering. No GL-only
work item is currently open; see [`todo.md`](todo.md) for cross-backend TODOs.
