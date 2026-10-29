# GL-Family Backend Design

`crates/gl` implements the Core backend SPI for three API profiles: desktop
OpenGL through WGL, OpenGL ES through EGL, and browser WebGL2. They share a
statically dispatched RHI implementation and operation lowering, while context
creation, native calls, capabilities, and browser lifetime are selected by
profile-specific providers. This gives callers one typed Core contract without
pretending the three APIs have identical feature sets.

## Architecture and layers

```text
Core API (`fluxel_rhi_core::api`)
        |
   GlInstance / Gl (backend/rhi)
        |  Core SPI adapter, typed resources, immediate command encoding
        +-------------------------------+
        |                               |
 shared GL-family operations       profile providers
 backend/rhi/{command,             native/{wgl,egl,discovery,
 resource,pipeline,shader,          provider,probes}
 completion,surface}                    |
        |                         browser/{owner,encoder}
        +---------------+---------------+
                        |
       context state, GL dispatch, driver / WebGL runtime
```

- **Public adapter (`backend/rhi`)** implements Core instance/device/resource,
  encoder, queue, surface, mapping, and completion traits. It validates Core
  identities and translates typed operations into GL work.
- **Operation lowering (`backend/rhi/command`, `raster`, `pipeline`,
  `resource`, `shader`)** handles transfer commands, raster and compute state,
  bindings, format/view choices, and shader compilation. Shared modules keep
  operation meaning aligned across profiles; branches are guarded by profile
  facts and capability admission.
- **Context and applied-state layer (`backend/state`, `backend/api`)** owns the
  context stamp, object slot/generation tracking, dispatch access, cached GL
  state, and the distinction between desired and driver-applied state. The
  cache avoids redundant calls; it is not a hazard tracker or resource-state
  model. Unknown state after interop, deletion, or loss is invalidated and
  rebound as needed.
- **Native providers (`backend/native`)** open and own WGL or EGL contexts,
  load entry points for that exact context, discover version/extensions/limits,
  and implement native drawable presentation. WGL and EGL details stay out of
  shared command lowering.
- **Browser provider (`backend/browser`)** owns WebGL2 objects and canvas
  access. It uses browser bindings rather than exposing a native dispatch
  pointer and implements browser-specific command encoding and readback.
- **Profile/feature translation (`backend/profile`, `features.rs`,
  `translate*.rs`)** maps observed profile facts into Core capabilities and
  translates raster/shader conventions. A feature is reported only when the
  selected context satisfies its version, extension, limit, and callable
  entry-point requirements.

These boundaries are visible in the source tree under
[`src/backend`](src/backend), with the public construction entry points in
[`src/lib.rs`](src/lib.rs). The module names above describe responsibilities;
individual implementation files are split further by operation family.

## Construction and runtime flow

1. **Select a provider.** The integration layer enables one Cargo profile
   feature and constructs `GlInstance`: WGL adopts a host window and opens a
   desktop core context; EGL adopts a host drawable or creates a pbuffer;
   WebGL2 creates a context for the supplied canvas. Native handles remain
   borrowed from the host for the lifetime documented by the unsafe
   constructors.
2. **Discover the exact context.** With that context current, the provider
   loads dispatch functions and records its actual API version, profile,
   extensions, limits, renderer identity, and supported operations. Requested
   version attributes alone do not establish support.
3. **Create typed Core objects.** The instance exposes an adapter/device using
   the discovered capability snapshot. Buffers, textures, views, samplers,
   pipelines, and bind groups carry the context/device identity and generation
   used to reject cross-device or stale use before issuing GL calls.
4. **Encode and execute immediately.** Creating an encoder does not create a
   portable deferred command buffer. Each accepted command validates its
   inputs, makes or checks the owner context current, lowers the operation,
   binds the needed state, and calls GL/WebGL on that thread. Submission
   publishes ordered-work/completion bookkeeping for work already issued to
   the driver; it does not replay the commands elsewhere.
5. **Observe results and present.** Completion polling remains non-blocking
   and owner-thread-affine. Fence observation, query availability, GPU staging
   copies, and CPU mapping are distinct steps. Surface presentation swaps the
   borrowed native drawable or publishes through the browser canvas contract.
6. **Resize, suspend, or lose the context.** Surface generations invalidate
   old acquired images. WebGL context loss terminates the current backend
   generation; after browser restoration the host constructs a new instance,
   device, and surface. Old resources are not revived by reusing their numeric
   GL names.

RenderGraph owns ordering and hazards. GL performs the caller-provided ordered
operations and validates backend-local identity and state; it does not infer
resource transitions or schedule independent GPU queues. Immediate execution
also means callers must obey context-thread affinity for operations and final
resource drops.

## What is distinctive about this backend

- **Immediate, context-thread execution.** Unlike DX12/Vulkan/WebGPU/Metal
  command encoders, GL command methods reach the driver as they are called.
  The context is not hidden behind a worker queue or a global lock.
- **One implementation, three different profiles.** Desktop GL, GLES, and
  WebGL2 reuse typed operation families, but feature admission is per-context.
  Desktop extensions cannot silently enable a route on GLES or WebGL2.
- **Explicit state knowledge.** The cache tracks what the backend believes is
  applied to the current context. Scoped native interop restores required
  state and invalidates uncertain cache entries on exit or unwind. WebGL2
  intentionally has no native dispatch-table pointer.
- **Names are not identities.** Native GL names can be recycled. Resources
  are tracked by context identity plus slot/generation so stale handles cannot
  accidentally refer to a newly allocated object.
- **No implicit GPU-wide stall.** Completion and mapping use supported fences,
  queries, and staging paths. `glFinish`, per-draw error draining, and hidden
  synchronous readback are not used as shortcuts.
- **Shader routes are profile-aware.** WGSL is reflected and lowered to the
  selected GLSL profile with binding and coordinate adaptation. Desktop GL
  SPIR-V is an independent artifact route; Vulkan SPIR-V is not treated as a
  portable GL shader binary.

## Context and resource lifetime

A native `GlInstance` owns the context and its dispatch/provider state while
borrowing the host drawable. The context must stay current on its owner thread
for calls and final native destruction. WebGL2 owns browser context-side
objects but the host owns canvas lifecycle and must notify context loss. The
backend keeps native object references behind typed resources; applications
must not retain raw numeric names as durable identifiers.

Resize changes drawable extent and surface generation, not device identity.
Context loss is stronger: the current browser generation becomes terminal,
leases and pending waiters are invalidated/woken, and restoration requires a
new backend generation. This distinction prevents old surface images or
resources from being used against a new context.

Scoped native interop is limited to native WGL/EGL builds. Its guard borrows
the dispatch table while the context is current, applies the documented
thread-affinity and unsafe rules, and invalidates backend state knowledge when
control returns. It does not hand out a command recorder or make external GL
objects part of the RHI resource identity system.

## Profile-specific behavior

- **Desktop GL4+/WGL:** WGL bootstraps a context, requests descending core
  versions, checks the actual created profile, then discovers exact function
  availability. Optional routes include GL SPIR-V and selected multiview,
  query, persistent mapping, and indirect-count paths when their complete
  requirements are present.
- **GLES3+/EGL:** EGL chooses a usable GLES 3.x context and discovers that
  context's extensions and limits. GLES has separate gates for persistent
  mapping, storage-buffer indexing, depth/stencil transfers, GPU-counted draws,
  and exact query/readback operations.
- **WebGL2:** browser context ownership and WebGL validation rules apply.
  Runtime binding arrays are specialized to bounded native slots; mapping and
  readback use fence-backed staging/snapshot flows. Standard WebGL2 has no
  compute route and does not expose compressed-block readback.
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
