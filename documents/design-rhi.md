# Fluxel RHI architecture and design rationale

This document describes the current source architecture and its reasons. The
public contract lives in [`core::api`](../crates/core/src/api/mod.rs); support
queries and native implementation determine which operations a device can
execute. API vocabulary alone is not an implementation or conformance claim.

## Overall design

### RHI is the execution boundary

Fluxel separates decisions about GPU work from the mechanisms that execute it.
RHI accepts resources, ordered operations, explicit synchronization intent, and
surface actions. It lowers them to a selected native API and reports acceptance,
failure, presentation, and completion facts. Keeping that boundary below the
RenderGraph avoids two competing schedulers or resource-state planners.

| Owner | Architectural responsibility |
| --- | --- |
| Host and platform adapter | Window/canvas ownership, borrowed native handles, event delivery, resize, foreground/background, and restoration policy |
| RenderGraph and Renderer | Resource usage planning, hazards, execution order, frame pacing, and retirement decisions based on completion evidence |
| Assets | Persistent resource identity, typed asset handles, caching, and CPU/GPU residency |
| RHI Core | Shared operation meanings, typed handles, device identity, capability contracts, and backend execution interfaces |
| RHI backend | Native resources, encoding or immediate execution, synchronization mechanisms, surface leases, and native in-flight ownership |

The host does not depend on RHI. A host-to-RHI adapter borrows a window handle
and establishes a surface. Browser and Android integration similarly connects
platform events to backend operations without moving GPU implementation into
the platform layer. Native window lifetime and GPU completion remain separate
obligations.

### Dependency and dispatch structure

The workspace has seven crates. Each backend depends on Core; the facade
reexports Core and enabled backends. Core does not depend on a backend, a host,
a renderer, or an application async runtime.

```text
Application / Renderer / platform integration
                    |
             fluxel-rhi facade
              /           \
             v             v
           Core <------ selected backend(s)

Backends: DX12, Vulkan, WebGPU, GL family, Metal
Arrows show dependencies; every backend implements Core's contracts.
```

`Device<B>` and its resources keep the backend type through execution. This
lets each backend retain native representation and thread-affinity constraints
without a mandatory backend enum, downcast registry, or global `Send + Sync`
requirement. Applications that need runtime backend choice perform that choice
at their integration boundary. Per-device capabilities remain dynamic: two
devices of the same backend can expose different limits and routes.

Core associates resources with a `DeviceId`; adapter, lane, plan, and step
identities express narrower ownership relationships. Type parameters prevent
cross-backend mixing, while identity checks reject cross-device mixing within
one backend. Unsafe native imports and execution bridges still require the
caller to uphold their lifetime and synchronization invariants. The typed API
does not substitute for RenderGraph hazard analysis.

### Two execution models, one operation meaning

DX12, Vulkan, WebGPU, and Metal record native commands through the deferred
backend contract. GL4, GLES, and WebGL2 issue native calls during encoder
operations through `ImmediateBackend`. The distinction is in trait availability:
GL does not implement deferred submission, and deferred backends do not acquire
an immediate mode merely for interface symmetry.

A GL encoder remembers pass state, bindings, and execution facts. It does not
collect a Rust command log for replay on another thread. Finishing an immediate
session reports work already issued; it cannot roll that work back. Deferred
recording also does not imply parallel recording support: that is a separate
capability. Browser thread affinity and GPU command recording are independent
properties, which is why WebGPU and WebGL2 use different models.

### Plans, failure, and completion

[`SubmissionPlan`](../crates/core/src/api/submission.rs) is a reusable ordered
execution description produced by a caller that has already resolved graph
dependencies. Its steps identify lanes, work kinds, and waits on preceding
steps. Its unsafe compiled-plan boundary requires a realizable layout; RHI
does not sort another DAG or infer resource hazards. A backend may reject plan
shapes that its native path cannot execute.

Completion selection belongs to each execution, independently of plan topology.
A completion point is host-observable evidence; it is not interchangeable with
a GPU queue wait. Receipts expose accepted steps and completion coverage by
lane so that the caller can reason about work actually issued.

Execution is not an atomic transaction. A failure after a native call may leave
an issued prefix, or native acceptance may be uncertain. Deferred and immediate
outcomes preserve those distinctions instead of reporting a clean rejection
after irreversible effects. This prevents unsafe reuse or retirement based on
an error that conceals in-flight work.

Backends retain native objects needed by accepted work until native completion
or a terminal failure establishes their disposition. This safety floor is
separate from the Renderer's resource retirement policy and from unsafe caller
obligations. Returning from encoding, presenting an image, or dropping a
completion observer does not establish GPU completion. Observation uses futures
without imposing a particular executor; context-affine futures can remain local.

### Surface images are exclusive leases

[`Surface`](../crates/core/src/api/surface.rs) separates the host-owned drawable
from a device-owned presentation path. An acquired image is an exclusive lease
for one surface generation, rather than an ordinary reusable texture view.
Presentation or abandonment consumes that ownership relationship.

Configuration and generation publication form one native transaction. A failed
or cancelled configure must not publish a half-configured generation, and
acquire observes the generation after native acquisition. This closes resize
and acquisition races without transferring window ownership or frame pacing
to RHI. Backend lease validation rejects stale use after reconfiguration or loss.

## Crate and backend designs

### `crates/core`: common semantics and the trust boundary

[`api`](../crates/core/src/api/mod.rs) contains both caller-facing types and
backend SPI, divided by concern: creation, shaders, passes, execution,
synchronization, surfaces, mapping, and optional operation domains. Sharing the
descriptors and result meanings avoids translation between subtly different
definitions of the same operation.

Core handles wrap backend-native objects and add identity and structural
validation. Native representation remains an associated backend type. Resource
creation consults support before calling the native constructor; optional
domains such as compute use typed capability access. An unavailable domain has
an explicit representation instead of a successful no-op implementation.
Structured errors distinguish unsupported work, invalid identity/state/ranges,
device loss, and surface lifecycle failures.

Core leaves native ownership machinery private. Browser tokens, GL contexts,
COM pointers, Vulkan handles, and Metal objects do not become portable resource
concepts. The unsafe SPI is the boundary at which a backend proves native
execution and lifetime invariants; Core supplies common checks without
pretending that all GPU correctness can be established locally.

### `crates/rhi`: a thin integration facade

The [facade](../crates/rhi/src/lib.rs) reexports `api` and feature-selected
backend modules, with target guards for platform-specific implementations.
It introduces no second resource model or runtime dispatcher. Consumers use a
single public namespace while choosing dependencies and backend types explicitly.

The numbered examples live here because they exercise the integration boundary:
the same scene and maintained WGSL source should exercise different execution
implementations. Host adapters, browser entry points, and capture helpers are
example integration code rather than Core policy. Native shader artifacts are
generated inputs, not separately maintained scene implementations. Explicit
visual fallbacks demonstrate a scene on a smaller capability set without
claiming that the backend implements the missing feature.

### `crates/dx12`: explicit native command lists and fence ownership

The [DX12 backend](../crates/dx12/src/backend.rs) lowers directly to native
resources, descriptors, pipelines, command lists, and a direct queue. Descriptor
binding namespaces agree with the DXIL shader ABI: bind groups map to register
spaces and bindings to registers. Keeping that agreement explicit avoids an
invisible remapping layer between shader artifacts and resource layouts.

The encoder translates supplied transitions rather than maintaining a global
resource-state history. Generic copies consume the caller's copy states;
convenience uploads have their own narrower state contract. Planning stays in
the caller while temporary upload resources remain backend-owned.

The [submission path](../crates/dx12/src/backend/encoder/submit.rs) serializes
native execute/present/signal operations and retains allocators, referenced
objects, and uploads through fence completion. A signal failure after execution
cannot be treated as unissued work. Device removal is also distinct from a
successfully reached fence value. These distinctions determine safe lifetime
release and prevent a success-shaped queue abstraction from concealing loss.

Runtime shaders consume DXIL. The examples' WGSL-to-DXIL toolchain is offline;
it does not add runtime WGSL compilation. DX12's native raster conventions
already match the public coordinate contract, making it a useful scene baseline
without making DX12-specific behavior part of that contract.

### `crates/vulkan`: native command buffers with explicit queue constraints

The [Vulkan backend](../crates/vulkan/src/backend/mod.rs) retains instance and
device ownership through native child objects. Device creation selects an actual
graphics queue and enables queried native features. The current raster path
uses dynamic rendering, avoiding a second persistent render-pass compatibility
model behind the public attachment descriptions.

Command buffers record supplied operations directly. A negative-height viewport
and corresponding front-face lowering preserve the common framebuffer
convention. Runtime shaders consume SPIR-V; generated example binaries preserve
the maintained WGSL source without introducing a runtime compiler fallback.

Submission uses distinct serialization for execution-plan ordering and Vulkan's
native queue external-synchronization requirement. These locks satisfy native
ordering obligations; they do not constitute a task scheduler. Fence-backed
in-flight ownership preserves accepted work through completion. The current
single-queue route rejects unsupported plan shapes and partial-tail resume
instead of presenting an emulated multi-lane implementation.

Descriptor layout creation uses the caller's explicit runtime-array capacity.
This is an operational allocation bound, not the largest integer a driver can
report as a limit. Matching Core validation and the actual native layout avoids
turning a broad capability limit into an unusable descriptor allocation.

### `crates/webgpu`: browser ownership with deferred native encoding

The [WebGPU backend](../crates/webgpu/src/backend/mod.rs) keeps JavaScript
objects on their browser owner thread. It records real `GPUCommandEncoder` and
pass encoder operations into one-shot command buffers, then submits them to
`GPUQueue`. Thread-local ownership does not require immediate GPU execution or
a CPU replay representation.

Native JavaScript objects referenced by submissions remain in flight until the
queue completion promise settles. Dropping a Rust observer does not cancel
submitted work or release its execution dependencies early. Promise completion
covers a native queue prefix; it does not provide an arbitrary cross-lane
timeline semaphore. The supported plan subset reflects that native mechanism.

The browser manages native resource transitions, so explicit public usage intent
does not justify adding a backend-wide hazard planner. WGSL goes to the native
WebGPU shader route. Validation failures and device loss remain observable
failures rather than reasons to silently retry using another backend.

A canvas current texture participates in the surface lease/generation contract.
DOM ownership, CSS size, animation scheduling, and restoration strategy stay
with the host. Adapter acquisition can fail in a particular browser environment;
the application must then choose another backend explicitly, rather than
turning `Device<WebGpu>` into a hidden GL implementation.

### `crates/gl`: shared semantics across three context profiles

The [GL-family implementation](../crates/gl/src/backend/gl/mod.rs) shares one
typed `Gl` backend across desktop GL, GLES, and WebGL2. Actual version,
extensions, and callable entry points jointly determine support. Shared
implementation avoids three public APIs, while profile-specific admission
prevents desktop features from leaking into GLES or WebGL2 capability answers.

The implementation separates typed GL-family operations, private native
providers, Core SPI adaptation, and state knowledge. Encoders issue calls on
the context owner thread. They neither dispatch to a command worker nor record
a deferred Rust command stream. Native context ownership can use local
borrowing and `Rc` without a mutex that falsely suggests cross-thread GL access
is valid.

The native providers' state mirror records what state is known to be applied. It begins
unknown and invalidates affected domains after uncertainty, external mutation,
resource deletion, or context loss. Context/slot/generation identity prevents
reuse of a numeric GL object name from resurrecting stale cache entries. This
is a binding/state optimization, not a resource hazard graph.

For example, attachment clears depend on native write masks as well as scissor.
The native provider establishes the masks needed by the clear, then invalidates pipeline
knowledge so that a subsequent bind restores the requested masks. Remembering
only the last requested pipeline would incorrectly suppress that restoration.
Cached intent and established native state are different architectural facts.
The browser's direct Core encoder establishes its required state separately;
it does not use this native-provider mirror. Shared semantics do not imply an
identical state-cache implementation across providers.

WGSL is lowered through Naga to the admitted GLSL profile, with reflected
bindings and backend coordinate adaptation. Texture sampling, texel access,
storage access where supported, and copy row order must agree on the upper-left
texture convention. Backend adaptation avoids per-scene flips that would
diverge across APIs.

#### Desktop GL4 / WGL

The WGL provider owns the current-context relationship and borrows the native
drawable for presentation. Desktop extensions can expose capabilities beyond
GLES, but only when the required native route is available. Immediate execution
and context affinity remain unchanged by the desktop profile's larger feature
set. Native fences establish completion independently of buffer swapping.

#### GLES / EGL

EGL owns a different context/drawable boundary while reusing GL-family lowering
and state semantics. GLES limits and extensions are queried for the actual
context, including Android contexts. This avoids both a lowest-common-
denominator public API and accidental inheritance of desktop-only features.
Android window and foreground/background events belong to platform integration;
the backend handles the resulting drawable and device operations.

#### Browser WebGL2

WebGL2 uses browser-thread JavaScript objects and the immediate contract. The
browser implementation includes a typed GL-family provider and a Core
`BrowserOwner`/`BrowserEncoder` execution path; they have distinct native call
boundaries and need their own validation. Neither is a deferred CPU command log.

WebGL2 has no native compute or secondary command recording route. Example 13's
historical `multithreaded_recording` name does not change this: its WebGL2 scene
uses sequential drawing. A finite capture may intentionally stop after frame
export; a stationary capture alone does not establish a blocked GPU.

One `GlInstance` represents one context generation. The host forwards loss via
`notify_context_lost()` and calls `preventDefault()` on the browser loss event
when restoration is desired. Loss makes the old device terminal, invalidates
old leases/resources, and wakes observers with `DeviceLost`. After restoration,
the host creates a new instance, device, and surface, then reconstructs resources
from CPU-side data. Restoration is not resurrection of old native objects.

Canvas backing-size changes require surface reconfiguration before acquisition;
out-of-band changes produce `SurfaceOutdated`. Animation callbacks and browser
presentation are not GPU completion evidence. Fence polling yields to the owner
thread's async execution. Internal lifetime fences may exist even when the
caller requests no public completion checkpoint: native safety and optional
host observation are separate requirements.

### `crates/metal`: direct encoding with a bounded binding ABI

The [Metal backend](../crates/metal/src/backend.rs) uses a native device, command
queue, command buffers, and pass encoders. Native tracked resource ordering
implements supplied execution intent without introducing a custom command
spine or graph. Completion handlers settle observations; dropping an observer
does not cancel issued work. Explicit retention covers dependencies such as
indirect argument resources where encoding alone is insufficient.

The shader route lowers WGSL through Naga to MSL and invokes the native compiler;
direct Metal source is also supported. A fixed binding ABI maps scalar bindings
using `7 * group + binding`: four groups occupy buffer slots 0–27, leaving slots
28–30 for vertex buffers. The disjoint partition prevents vertex input from
overwriting resource bindings. Arrays and layouts outside the implemented ABI
are rejected rather than hidden behind an unimplemented argument-buffer model.

The current submission subset is graphics-lane command steps with optional
presentation, without partial-tail resume. The source contains a `CAMetalLayer`
surface path. This is an implementation boundary, not completed platform
acceptance: macOS SDK linking, real Apple GPU execution, and example visual
review remain unverified. Other backends' evidence cannot close that gap.

## Shared observable semantics and their reasons

### Coordinates belong to the backend adaptation

The public convention uses NDC X/Y in `[-1, 1]`, Y upward, and Z in `[0, 1]`.
Framebuffer and viewport coordinates start at the upper-left and increase
downward in Y; `FrontFace` describes framebuffer winding. Texture texel `(0, 0)`,
upload row zero, and readback row zero refer to the upper-left row. Normalized
texture `v = 0` addresses the upper edge.

Backends adapt native viewport, winding, depth, shader access, and copy lowering
to these meanings. Keeping adaptation there lets the same shader and scene
express the same image across APIs. World handedness, camera direction, matrix
storage, and multiplication order remain application shader ABI decisions.

### Shader language and feature vocabulary are not universal support

[`ShaderDesc`](../crates/core/src/api/shader.rs) describes the supplied artifact;
route selection distinguishes direct native input, supported Naga lowering, and
unsupported input. A selected compiler's failure is reported rather than retried
through another route that may interpret the program differently. Maintained
WGSL examples and generated native artifacts are a source-management decision,
distinct from each backend's runtime format support.

The same rule applies to compute, mesh, ray tracing, sparse resources, queries,
indirect commands, and secondary recording. Public types make native operations
expressible; only implemented per-device routes establish support. Descriptor-
specific limits remain checked at creation or execution. Capability answers
support workload selection rather than merely describing what the upstream
native API could theoretically do.

### Binding bounds are caller intent, validated against native limits

`BindingCount::RuntimeSized { max_count }` supplies a nonzero layout capacity.
A bind group supplies a nonempty prefix within that bound; shaders may access
only supplied elements, and partially bound holes require explicit support.
Core and native layout creation validate the same bound. Capacity belongs to
the caller's workload, rather than being inferred from a driver maximum or
turned into a Core allocation policy.

Binding arrays have exactly one dimension: present elements are concrete
resources, including when the layout allows holes. Nested arrays, even empty
ones, are rejected before native allocation. Resource provenance checks then
apply to every supplied element. Acceleration inputs follow the same ownership
rule: sizes, creation, and build all validate geometry buffers and referenced
BLAS identities before calling the ray SPI.

Dynamic buffer offsets follow ascending binding number, independent of layout
entry order or shader visibility. Each is added to the bound range start;
alignment, overflow, and allocation bounds are validated without changing the
range size or minimum binding size. This gives one binding ABI across native
descriptor schemes rather than exposing each driver's offset ordering.

### Copies, mapping, and queries preserve precise evidence

Generic copies require caller-established `CopySrc`/`CopyDst` states and leave
those states unchanged. The caller supplies subsequent transitions. Convenience
uploads own temporary upload transitions; on DX12 their destination starts and
ends in `COMMON`. Pass-local resolve and surface presentation retain their
native transition requirements. These local contracts avoid a hidden global
state tracker while keeping temporary transfer mechanics inside the backend.

Mapping requires the caller to establish completion and absence of incompatible
GPU access. A map lease controls CPU access and unmapping; cancellation must not
strand native mapping state. Mapping availability is not evidence that earlier
GPU work completed unless the operation's contract supplies that evidence.

Occlusion query results guarantee zero for no samples passed and nonzero for
visibility. Exact positive values can represent native counts or boolean
visibility and are not comparable across backends. This preserves the useful
portable fact without inventing precision that a native route cannot provide.

The query domain retains its selected kind; query sets retain kind and slot
count. Core rejects mismatched creation/operations, zero counts, and empty,
inverted, or out-of-bounds ranges. Raster-pass query declarations also validate
kind, device, and timestamp slots before native recording. Backends remain
responsible for supported query routes, active-query sequencing, resolve layout,
alignment, and native limits. Retaining this small immutable metadata gives
callers consistent input validation without creating a query scheduler in Core.

Both ordinary queue and multi-queue submission validate the same work packet
before native acceptance: step count/order, payload ownership and kind, lane,
surface lease, and completion selection. Multi-queue signal/wait validation is
additional to this shared integrity check. Invalid input returns the original
owning packet as a clean rejection; native partial/uncertain acceptance remains
the backend's responsibility.

## Verification and extension boundary

The [testing guide](testing.md), [conformance matrix](conformance-matrix.md),
[example instructions](../crates/rhi/examples/README.md), and
[Android evidence packet](evidence/android-2026-10-20/README.md) record verification
scope separately from architecture. The Android packet covers the tested Vulkan,
GLES, and WebGL2 example routes; WebGPU adapter acquisition was unavailable in
that environment. It does not imply universal Android driver support or close
renderer-owned restoration integration.

Compilation, successful submission, and native validation alone do not prove
correct GPU output or presentation. A new route needs consistent public support
answers, native creation/execution/lifetime behavior, focused real-GPU output
checks, and surface/visual evidence where applicable. Extending a backend must
preserve these ownership boundaries; a consumer fallback or another backend's
successful run cannot stand in for its missing native implementation.
