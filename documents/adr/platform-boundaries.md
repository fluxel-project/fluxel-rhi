# Backend and platform boundaries

This file groups related decision records to keep the index compact. Original ADR identifiers and status labels are retained; current normative contracts live in the linked design and testing documents.

## ADR-0002: Contain native handles and unsafe in the RHI implementation

**Status:** Accepted

## Context

DX12 and Vulkan lifetime, state, alignment, and queue rules cannot safely be
represented by leaked HAL handles in application-facing APIs.

## Decision

Keep HAL/native types and backend `unsafe` within each backend crate's private
native boundary. The public Core API exposes opaque, device-affine resources,
pipelines, bindings, leases, and structured errors. A few documented unsafe
Core constructors and accessors form the backend-only bridge; callers must
uphold their device identity and exclusive-lease contracts.

## Alternatives

- Expose raw handles for renderer or graph code to record commands.
- Spread small `unsafe` wrappers across graph and renderer modules.

## Consequences

The backend must validate device identity, usage, state, ranges, alignment,
thread serialization, and lifetime before native calls. Renderer and
RenderGraph can state portable contracts without inheriting HAL safety rules.

## Evidence

0.1.0 established the boundary; 0.1.1 resource leasing and 0.1.2–0.1.4 native
execution validated it. 0.2.8 preserves it while splitting the private module.

See the [workspace architecture](../../crates/rhi/DESIGN.md#system-boundary) for the current native boundary.

## ADR-0011: Build the GL family through three private layers

**Status:** Accepted

## Context

Desktop GL, GLES, and WebGL2 share a stateful execution model but have
different context creation, core versions, extensions, and entry points. A
single public browser/session or raw-context model would leak platform details
and duplicate resource ownership.

## Decision

`backend/gl` has three private responsibilities:

1. Core `api` defines the portable immediate-execution contract and structured
   failures; the GL backend selects the available profile and feature routes;
2. `state` remembers known/unknown driver bindings and invalidates derived
   cache facts when dependencies change;
3. native/browser providers own contexts and issue GL calls directly while
   lowering each Core command. They must not buffer a CPU command stream for
   later replay.

Desktop GL, GLES, and adopted WebGL2 create separate backend-private providers
and terminal device identities. A capability exists only when the selected
core-or-extension route and every required function entry point are present.
No raw context, extension object, browser session, or GL name enters public
RHI. Context loss is device loss.

## Consequences

The backend can use the highest available GL 4.x or ES 3.x context while
retaining the GL 4.0 / ES 3.0 / WebGL2 floor. State caching records native
state, not commands: cache misses, raw access, retirement, and loss must
invalidate facts conservatively. Runtime evidence remains required per
provider route and belongs in conformance results, not this ADR.

## ADR-0012: Keep RHI limited to portable execution

**Status:** Superseded by the current [RHI architecture](../../crates/rhi/DESIGN.md).

The `command::ResourceUse` collection and backend hazard derivation described
below are historical. RenderGraph plans hazards and resource states; the RHI
executes explicit commands and transitions without collecting uses or building
a second dependency graph.

> Clarification (post-foundation target): this ADR keeps RHI free of
> RenderGraph-owned declarations and renderer policy. It does not require
> RenderGraph to be a pure IR with no RHI dependency. RenderGraph directly
> reuses the portable RHI contract; only backend-private APIs remain outside
> RenderGraph.

## Context

RenderGraph scheduling, declared coverage contracts, and renderer policy have
different consumers and lifetimes from portable GPU execution. Putting them in
RHI made backend implementations learn upper-layer concepts they cannot lower.

## Decision

RHI records only actual command-derived `command::ResourceUse`, `RecordedWork`,
`SubmissionPlan`, plan/completion points, and transient lifetimes. It has no
graph bridge, declared-work contract, or external declared-use validation.

`ResourceUse` includes synchronization-only objects when a command has a real
producer/consumer relation but no portable byte-addressable backing. Query-set
slot writes and query resolves are the current case: they retain set identity,
slot range, stage scope, and query access in the work record. They are neither
invented buffers nor native barrier commands.

## Consequences

RenderGraph may keep its own declarations and scheduling checks. Backends see a
small execution vocabulary and derive native hazards from actual uses. New
upper-layer policy must not expand the RHI object model.

A plan can therefore derive an ordering edge between a query producer and a
resolve on another lane before any backend chooses a query-pool, fence, or
memory-dependency mechanism. Query-only uses do not imply a buffer/image state
transition by themselves.

## ADR-0023: Keep host windows separate from RHI presentation integration

**Status:** Accepted

## Context

The host owns native window and canvas lifetimes and reports lifecycle events.
The RHI owns surface configuration and the exclusive acquired-image lease.
A copied `HWND`, `ANativeWindow*`, `UIView*`, or browser canvas handle cannot
extend the host object's lifetime.

## Decision

Keep the host and RHI ownership domains distinct. Platform integration code
borrows the live host handle to create a backend `Surface`, and keeps the host
object alive until that surface and all its `SurfaceImage` leases retire.
Backend-specific native handles and surface registration stay behind the
backend boundary. Core exposes `Surface`, `SurfaceImage`, and `PresentBinding`
without depending on a host or browser session type.

The current Windows examples put this integration in `examples/utils`; the
DX12 and Vulkan constructors explicitly mark raw-window-handle lifetime as an
unsafe caller obligation. WebGPU accepts a browser canvas through its backend
constructor. These routes are platform integration code, not another RHI
execution domain, resource manager, or RenderGraph.

On host destruction, integration code must retire the surface and its image
leases before releasing the native host object. A stale acquired-image
generation must reject present handoff through the surface contract.

## Consequences

- A platform route must document how it keeps the host target alive through
  surface use and how resize or destruction invalidates outstanding leases.
- Native host setup belongs in the platform integration or test fixture; the
  portable workload uses Core RHI operations once it has a surface.
- Backends can use different native presentation objects without adding host
  window types to the Core API.
