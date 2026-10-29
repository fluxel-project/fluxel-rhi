# Metal Backend Design

`fluxel-rhi-metal` is the statically dispatched Apple backend. It turns the
portable Core contract into Metal resources, command buffers, encoders,
completion facts, and `CAMetalLayer` presentation. It is intentionally not a
renderer: the caller supplies pass order, hazards, resource transitions, frame
pacing, and retirement policy; this crate validates and executes that plan.

The public surface is deliberately small: `Metal` is the backend type,
`MetalInstance` discovers devices, and `MetalSurface` binds a host-owned
`CAMetalLayer`. All Objective-C and Metal objects remain private to this crate.

## What is specific to Metal

Metal has a deferred, command-buffer execution model. A Core encoder call is
lowered immediately into an `MTLCommandBuffer` and one active native render or
compute encoder. The backend does not first build a second portable command
list, and it does not replay application work on another CPU thread.

The normal path has one graphics lane. Where the device and requested packet
admit it, the multi-queue path exposes a compute lane and uses Metal shared
events at explicit packet boundaries. This is a deliberately narrow mapping:
it does not infer dependencies between arbitrary passes or turn a Core packet
into a scheduler.

Resources use Metal storage modes that match the Core memory class: device
memory becomes `Private`; upload and readback memory become `Shared`. Mapping
is tracked per buffer, and resource, bind-group, and pipeline objects retain
their native owner through the typed Core object graph. The public Core device
identity remains the authority for rejecting cross-device misuse before an
Objective-C call is made.

## Layers and ownership

```text
Core API and submission plan
            |
            v
Core backend SPI -- device/provenance validation
            |
            v
Metal device layer -- MTLDevice, queues, capabilities, upload/cache state
            |
            +-- resource / binding / pipeline / shader modules
            |
            +-- command encoder -- MTLCommandBuffer + render/compute encoders
            |
            +-- submit + completion -- commit, terminal status, retention
            |
            +-- surface -- CAMetalLayer, drawable worker, generation lease
```

`backend/mod.rs` owns the native device, the graphics and compute command
queues, capability facts, queue serialization, lane frontiers, submission
tracking, upload pool, and helper caches. It is the only place that assembles
the backend's native object families. Submodules own narrower boundaries:

- `resource`, `texture`, `sampler`, `memory`, and `mapping` create and manage
  Metal resources and their CPU visibility rules.
- `bindings`, `binding_abi`, and `argument` map Core layouts into Metal's
  stage-local buffer, texture, and sampler namespaces.
- `shader`, `shader_variant`, `pipeline`, `compute`, `mesh`, and `ray` turn
  admitted shader and pipeline inputs into native state.
- `encoder`, `copy`, `query`, and `readback` lower individual operation kinds
  into native encoders. They do not decide pass order or resource lifetime.
- `submit` owns command-buffer commitment, completion observation, and the
  native objects that must remain alive until terminal completion.
- `surface` owns layer configuration and drawable leases, while the host keeps
  ownership of the view, layer attachment, event loop, and resize lifecycle.

This separation keeps Objective-C retain/release, callbacks, and unsafe native
calls inside their owning modules. It also prevents the RHI public API from
leaking `MTLDevice`, `MTLTexture`, `CAMetalDrawable`, or a host window object.

## Normal execution flow

1. `MetalInstance` enumerates Apple devices. Opening an adapter creates the
   native queues and records capabilities and limits from that exact device.
2. Core creates typed buffers, textures, views, samplers, layouts, shaders,
   and pipelines. Capability and format checks happen before native creation;
   unsupported combinations return structured errors.
3. Core begins encoding on a requested lane. The Metal encoder creates an
   `MTLCommandBuffer`, then opens and closes native render or compute encoders
   as the typed operations require. It holds only transient encoding state,
   native upload slices, query work, and objects that a command buffer needs
   to keep alive; it is not a portable CPU command representation.
4. `finish_encoder` rejects an encoder with an active native pass or the wrong
   lane. A finished recording contains the native command buffer and its
   in-flight retention set.
5. Submission validates Core step/lane constraints, commits command buffers
   in the caller's supplied order, and optionally commits the presentation
   handoff. A failure after any commit is reported as partial or indeterminate
   acceptance with completion evidence for the committed prefix.
6. Metal completion handlers publish terminal command-buffer success or
   failure. They release retained upload slices, indirect arguments, external
   resources, argument arrays, and acceleration structures only after that
   terminal state. Observing or dropping a completion future never cancels
   committed GPU work.

Core uses those completion facts to decide when its own resources may retire.
The backend never substitutes `waitUntilCompleted`, an implicit CPU readback,
or a renderer retirement policy for that contract.

## Binding and shader ABI

The scalar binding ABI assigns Core bindings to fixed native slots using
`7 * group + binding`. Buffer, texture, and sampler slots are independent
within each shader stage; push constants follow resource buffer slots and
vertex inputs occupy reserved upper buffer slots. Pipeline creation checks the
combined per-stage native budget after slot assignment. A layout that cannot
fit is rejected rather than silently routed through an unimplemented
argument-buffer layout.

Binding arrays are a separate optional path. Their support depends on the
selected device's argument-buffer capabilities. Array materialization uses
fresh native storage for each logical element so later CPU changes cannot
mutate an array already consumed by a submitted command buffer.

WGSL follows a two-part route. Naga parsing, validation, and MSL generation
run in a bounded CPU worker because they operate on `Send` Rust data and can
be expensive. Pipeline-specific stage-slot assignment then creates the native
library and pipeline on the Metal ownership path. Direct `MetalSource` and
precompiled `MetalLibrary` inputs are distinct routes. WGSL does not bypass the
layout ABI, and direct native shaders must use the same layout-derived indices.

## Synchronization, transitions, and queries

Same-queue work relies on Metal's normal command-buffer ordering while Core
supplies explicit transition intent. The encoder can use a narrow native event
boundary for untracked placement-heap or alias transitions; that mechanism is
not general hazard analysis and does not make arbitrary resource use safe.

The multi-queue implementation validates lane and step restrictions before
native handoff. It creates shared-event waits/signals only at the explicit
packet boundaries requested by Core, serializes queue construction where
needed, and reports frontiers through the normal completion machinery. Sparse
binding follows this path only when the native placement route is supported.

Timestamp and occlusion support are discovered as device capabilities. Query
publication, GPU readback/copy work, and CPU consumption are separate events.
The backend does not claim that a successful submit or a presentation call is
proof that the query result, pixels, or frame are complete.

## Surface lifecycle and presentation

`MetalSurface::from_layer` receives a retained, host-owned `CAMetalLayer`,
associates it with the selected `MTLDevice`, and configures bounded drawable
acquisition. The host still attaches the layer to a view and supplies resize,
suspend, resume, foreground, and restoration policy.

Configuration validates the complete Surface request before publishing a new
generation. Drawable acquisition runs through one bounded worker: merely
constructing or polling the public future never calls `nextDrawable`. If the
request is cancelled, its eventual result is discarded. If resize or
reconfiguration occurs first, the old generation cannot mint a usable image.

An acquired image is an exclusive, generation-bound `CAMetalDrawable` lease.
Its final handoff checks both Core's generation and the surface's current
generation under the same lock, then allows presentation once. A stale or
already handed-off lease returns a structured error. The drawable's texture is
kept alive through encoding and handoff.

## Capability scope

The implemented public surface includes graphics and compute pipelines, up to
eight render targets, depth/stencil, dynamic uniform and storage offsets,
direct and supported indirect draw/dispatch, clear/copy/resolve/blit paths,
mapping, placement heaps, binary archives, native interop, and format support
for BC, ETC/EAC, and ASTC where the selected Apple device and usage allow it.

Capabilities are facts from the opened device, rather than promises from the
crate name. Argument arrays and nonuniform indexing, mesh/object shaders,
acceleration structures and ray queries, multiview, query kinds, GPU-count
multi-draw, second-lane submission, sparse placement, external resources, and
HDR Surface options each have their own native check. Acceleration-structure
construction does not imply ray-pipeline tracing, and binary archives do not
provide a DXR-style state-object cache.

There is no Apple GPU execution or Surface presentation evidence in this
repository yet. Cross-compilation, static inspection, and unit tests establish
only source-level behavior; they do not close a real-device capability or
visual-presentation claim. The remaining cross-backend work and reopening
conditions are tracked in [`TODO.md`](../../TODO.md).

## Code map

- Device and Core SPI: `src/backend/mod.rs`
- Native command lowering: `src/backend/encoder/`
- Submission and completion: `src/backend/submit/`
- Surface and drawable acquisition: `src/backend/surface/`
- Resource and memory rules: `src/backend/resource.rs`, `texture/`,
  `memory.rs`, and `mapping.rs`
- Binding ABI and shader routes: `src/backend/binding_abi.rs`, `bindings/`,
  `argument.rs`, and `shader.rs`
- Device feature derivation: `src/backend/capabilities.rs`
