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

## Public feature coverage

### Desktop GL4+

Immediate graphics, MRT, depth/stencil transfers, standalone resolves,
matching-sample MSAA color texture copies, 1D sampled textures, Float64 vertex
attributes, direct shader constants, bounded runtime binding arrays,
GPU-staged readback, compute, and supported indirect commands are available.
Native limits gate nonuniform indexing, persistent coherent mapping,
GPU-count drawing, queries/statistics, program binary caches, conservative
rasterization, and contiguous OVR multiview.

### GLES3+

Immediate graphics, MRT, bounded runtime arrays, depth/stencil uploads and
copies, standalone MSAA resolves, GPU-staged readback, and supported compute,
nonuniform indexing, and indirect commands are available. Same-sample MSAA
texture copy remains unsupported pending complete runtime acceptance. ES 3.1
and extensions gate GPU-count lowering, sampled depth/stencil readback,
coherent persistent mapping, exact color readback, queries, program caches, and
contiguous OVR multiview.

### WebGL2

Immediate graphics, MRT, bounded runtime binding arrays, depth/stencil plane
transfers, copies/resolves, GPU-staged readback, asynchronous snapshot mapping,
and occlusion are available. Native limits bound array specialization and
exact format readback; combined nonuniform texture/SSBO support remains
unavailable. Extensions gate compressed textures and other optional
operations. Standard WebGL2 has no compute or compressed-block readback.

### Compressed textures

- **BC1–BC7:** Desktop and GLES use core/extension-gated upload and sampling;
  `COPY_SRC` depends on format and transfer route. WebGL2 requires an extension
  and has no compressed `COPY_SRC`.
- **ETC2/EAC:** GLES and WebGL2 upload/sampling depend on profile support;
  `COPY_SRC` depends on native copy and readback paths. WebGL2 has no
  compressed `COPY_SRC`.
- **ASTC:** GLES LDR/HDR routes and WebGL2 profile/extension routes gate upload
  and sampling. `COPY_SRC` depends on the exact format and transfer route;
  WebGL2 has no compressed `COPY_SRC`.

For the exact compressed format, dimension, and usage, query
`texture_support`. GLES additionally probes the actual compressed-to-integer
block copy per format; an exposed image-copy extension does not prove that
reinterpretation works. Ordinary texture-to-texture copies retain independent
support.

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
work item is currently open; see [`TODO.md`](../../TODO.md) for cross-backend TODOs.
