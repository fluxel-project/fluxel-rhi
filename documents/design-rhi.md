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

RenderGraph is the intended consumer. API ergonomics means preserving its
compiled execution structure with minimal conversion, not minimizing example
source length. Explicit steps, waits, lanes, and completion selection carry
necessary execution facts. Standalone frame helpers and termination policy
belong to the examples; they do not add scheduling or convenience APIs to Core.

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

### Common execution entry

`Device<B>::begin_execution(plan, completion)` opens the optional
`Execution<B>` interface for already ordered Commands/Present plans. All shipped
backend types select their native `ExecutionMode` through the separate
`ExecutionBackend` trait; existing `Backend` implementations gain no required
associated types. Calls remain statically dispatched. The interface does not
compile a graph, choose fallback passes, or retire resources.

| Operation | Deferred mode | Immediate mode |
| --- | --- | --- |
| `commands(step, encode)` | Invoke `encode` now, retain its native RecordedWork | Invoke `encode` now; each command calls the native API immediately |
| `present(step, binding)` | Retain the exclusive binding for submission | Perform native presentation handoff now |
| `finish()` | Submit the collected native work and bindings | Close the session and obtain requested completion evidence |

The closure is never stored. Deferred storage contains native recordings and
bindings, not a portable CPU instruction list. Immediate state tracks the plan,
cursor and terminal result; it adds no replay buffer, per-step fence, GPU idle
wait, or scheduler. Successful presentation facts are assembled from the
already known cursor at finish, not from per-command event objects.

The common shape intentionally preserves different irreversible boundaries.
`commands` success is not GPU completion, and deferred `present` success means
that the binding was stored, not that native presentation was accepted.
`PresentError` returns a binding only when handoff did not consume it. Callers
must finish even after an encoding error to obtain native effects and completion
or uncertainty evidence; early return with `?` must not release referenced
resources. Starting/finishing execution retains the explicit unsafe lifetime
contract of the underlying APIs. No host wait is hidden in this entry.

Both owning outcome types implement `ExecutionOutcome`: `facts()` borrows
reported observations and `error()` exposes a failure without backend branching.
The owning result preserves the original SubmitOutcome or immediate finish
result, including retryable native work and uncertainty. Its borrowed facts do
not clone checkpoints. `None` means a fact is unavailable, not an empty set.
Accepted steps, affected native work and GPU completion are distinct: an
immediate failure can affect a step without completely issuing it. The common
entry additionally knows the successfully closed prefix, so it does not publish
the failing step as accepted. Legacy outcome accessors remain conservative when
the old value does not contain enough information to identify that boundary.
Checkpoint coverage for affected work is a lifetime observation, not proof that
all intended commands in a failed step ran. `ExecutionStatus::Accepted` never
means the GPU has completed the work.

`ExecutionFacts::failed_step` additionally reports an identifiable failed slot
with `StepEffect::None`, `Partial`, or `Unknown`. The optional field being absent
means the source cannot identify a failed slot, or no slot failed; it does not
prove absence of native work. A known clean step rejection stays None even if
an emergency checkpoint for earlier work is indeterminate. Conversely, failure
to create completion evidence after all steps does not invent a failing step.
Legacy owning outcomes do not gain fields or lose recovery payloads; their
borrowed views remain conservative when they lack boundary information.

A clean immediate Present rejection returns its lease and permits retry. If
execution ends without a successful retry, its original rejection error and
no-effect slot remain observable. Successful retry clears that transient
rejection. This metadata introduces no command recording, native wait, or new
recovery policy. The newer facts/outcome structs gain the failed-step field;
external struct-literal constructors must initialize it, while the older Core
execution SPI and owning submission enums keep their signatures.

The entry rejects SparseBind steps before issue. Sparse operations, native
multi-queue tokens, parallel recording continue to use their existing explicit APIs.
`recorded_commands` is a Deferred-only binding operation on the common entry; it
accepts already recorded native work without conversion and returns ownership
on a validation failure. This deliberately bounds the common entry;
it does not remove advanced execution or make GL pretend to support deferred
recording. All 15 examples use this entry: GL issues the shared encoding closure directly,
while the deferred submission helper binds existing native recordings through
`recorded_commands`. Example 13 retains its parallel secondary recording.

### Ordered dependencies and visibility

`SubmissionPlan::validate_for_execution` checks device ownership, declared lanes,
and predecessor indices before execution starts. The common entry and legacy
ImmediateSession use this check; it does not sort steps or infer hazards.
Validation runs at execution initialization, not at each draw.

A lane intrinsically orders its own work. `LaneInfo::route_to` and
`Device::wait_route` therefore report SameQueue for a valid same-lane pair even
when its explicit cross-lane route table is empty. This does not advertise a
foreign-lane native wait. GL exposes one context lane: after Core validates the
whole ordered plan, same-lane StepWait dependencies need no extra native wait.
The wait descriptors still reach the backend SPI unchanged.

Ordering and resource visibility are separate obligations. Explicit
ResourceTransition operations still lower required memory barriers; a same-lane
wait neither invents those transitions nor makes unsupported visibility or
cross-lane paths available. The unsafe compiled-plan boundary continues to
require a natively realizable layout.

### Plans, failure, and completion

[`SubmissionPlan`](../crates/core/src/api/submission.rs) is a reusable ordered
execution description produced by a caller that has already resolved graph
dependencies. Its steps identify lanes, work kinds, and waits on preceding
steps. Its unsafe compiled-plan boundary requires a realizable layout; RHI
does not sort another DAG or infer resource hazards. A backend may reject plan
shapes that its native path cannot execute.

`PlanId` and `StepId` name reusable topology, not an individual execution.
Completion points and native signal tokens carry facts about actual submitted
work. A multi-queue signal's identity metadata is read-only after native creation;
when plans overlap, the caller must associate each token with its intended
producer and consumer executions. Static step equality cannot prove that
association. Core validates device, topology, lane, and route facts without
maintaining an execution registry or inferring cross-submission dependencies.
DX12 and Vulkan implement graphics/compute lanes with native cross-queue
signals and waits. Their focused GPU fixtures exercise compute producers and
graphics consumers; they do not prove every possible submission topology.
Metal conditionally exposes two native queues with shared-event waits inside
one packet, limited to four waits per recording. It rejects cross-packet GPU
tokens. The Metal route has Apple-target compilation and static review only,
without Apple GPU acceptance. Mock contract tests are not native conformance
evidence.

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

Example recording returns the native work and any pending resource-state commit
separately. The deferred host publishes that commit only after successful
acceptance and its deliberately serial completion wait. Partial or uncertain
acceptance does not permit reuse of speculative state: the example retains the
full outcome, drains issued work, and terminates the failed frame. A production
RenderGraph can instead use the accepted prefix and its own recovery policy.

Frame-local attachments and readback tickets stay in the scene through actual
completion. Immediate execution needs the same lifetime proof even though no
deferred submission occurs. Native host teardown drains before dropping the
scene. Browser examples explicitly drain on their normal exit; cancellation or
failure without completion proof traps without unwinding the in-flight scene.
The host must discard that failed WASM instance. A failed idle wait is not
completion evidence. This conservative example policy is not an RHI device-loss
or resource-retirement manager.

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
in-flight ownership preserves accepted work through completion. Devices with
the required queue families also expose native graphics and compute lanes;
timeline semaphore edges carry cross-queue dependencies, including sparse
bind completion. Unsupported plan shapes and partial-tail resume are rejected.

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

The device requests supported standard optional features and adapter limits,
including timestamp queries, compressed formats, subgroup controls, and native
immediate data where the browser exposes the matching WGSL language feature.
Shader specialization constants and stage-specific push data are lowered at
pipeline creation. The immediate lowering retains the original WGSL source and
compiles a selected-entry variant; it does not record application commands.

Uploads and copies use native copies where their alignment permits. GPU
repacking handles byte offsets and row pitches that cannot be submitted directly.
Depth uploads and single-aspect combined depth/stencil copies use native render
passes when WebGPU forbids the corresponding texel copy. Partial clears and
resolves preserve pixels outside the public render area. These private operations
retain their temporary resources through the same queue completion boundary.

GPU-count multi-draw snapshots the argument and count buffers on the GPU,
produces masked indirect arguments in a compute pass, and records the bounded
native indirect calls. A native raster pass can be split around this work,
preserving attachment contents, bindings, dynamic state and logical queries.
The state retained for reopening a pass contains no draw calls and is not a CPU
command stream. There is no CPU count readback or queue idle wait.

Logical occlusion slots can collect multiple native pass segments. Resolved
segments are added on the GPU, so recording order does not determine the value
seen by a later submitted resolve. Timestamp measurements use standard pass
`timestampWrites`, including auxiliary clear and resolve work. Mapping awaits
`mapAsync` and exposes owned Rust bytes rather than browser-managed pointers.

The standard browser API has no matching ray/mesh, sparse binding, independent
queues, multiview, full pipeline-statistics, native placement-heap or serialized
pipeline-cache interface. A render bundle also cannot express the independent
viewport/scissor operations required by the current secondary-recording contract.
Browser video external textures require a source/lifetime contract absent from
Core's ordinary texture import API. Depth24Plus has no defined portable texel
byte representation; compressed texture color clears have no standard native
clear or attachment path. Neither boundary is a claim that a separate software
codec or renderer could never provide an alternative.

Canvas configuration exposes standard format, usage, alpha and sRGB/DisplayP3
choices. HDR10 PQ and ExtendedLinearSrgb do not have matching native canvas
color-space declarations. Display conversion is not inferred from tone mapping
alone. Device feature checks remain necessary even when a format or operation
appears in the standard.

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

GL texture storage uses Core row order directly: logical row zero is native
texel row zero for ordinary and compressed textures. Uploads, image operations,
copies and readbacks therefore preserve row order. Rasterization reverses the
vertex clip-space Y coordinate and native winding together, while viewport,
scissor and fragment positions address these texture rows directly. Surface
passes use the same offscreen convention; import and publication reverse Y at
the default framebuffer boundary. A multisample resolve precedes that reversal.

This storage convention also preserves native cube-face adjacency. Reflecting
each face's sampling direction independently can match face interiors while
fetching the wrong neighboring face during seamless filtering. Real texture
views instead share the original storage, with independent admitted mip/layer
ranges and declared compatible formats. Copying a texture once does not provide
view aliasing semantics.

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
direct Metal source and precompiled Metal library bytes are also supported.
Library loading resolves and checks the requested entry point and stage; it
does not by itself provide a ray-pipeline or shader-table ABI. A fixed binding
ABI maps scalar bindings
using `7 * group + binding`: four groups occupy buffer slots 0–27, leaving slots
28–30 for vertex buffers. The disjoint partition prevents vertex input from
overwriting resource bindings. Arrays and layouts outside the implemented ABI
are rejected rather than hidden behind an unimplemented argument-buffer model.

The ordinary submission subset is graphics-lane command steps with optional
presentation, without partial-tail resume. A separate restricted native
graphics/compute multi-queue route is described below. The source contains a
`CAMetalLayer` surface path. This is an implementation boundary, not completed
platform acceptance: real Apple GPU execution and example visual review remain
unverified. Other backends' evidence cannot close that gap.

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

Standalone texture clear selects nonempty mip and array-layer ranges and a
color or depth value. It clears whole selected subresources; the caller supplies
the transition to color-attachment or depth-write state. The DX12 route requires
render-attachment usage and currently accepts float color clears for RGBA8 or
BGRA8 unorm and depth clears for Depth32Float. Other formats and stencil clears
are not covered. Standalone buffer clear remains unsupported by DX12. Render
pass attachment load clears are a separate operation.

External resource import is an explicit bridge contract. The bridge provides
an owned native handle and the producer's exact allocation and resource
metadata, and guarantees matching ownership transfers and synchronization.
An imported buffer or texture can be obtained for ordinary commands and
bindings. Texture transfers specify the subresources, usages and old/new
layouts. The producer's synchronization payload is imported before acquire;
release returns a payload signaled only when the recorded work is submitted
and reaches the release. The bridge must not use the resource outside the
matching ownership interval or wait twice on a binary semaphore payload.

Vulkan currently imports opaque Win32 handles or opaque file descriptors when
the relevant extensions and per-resource import features are available. Win32
handles are closed after import; successful FD imports transfer ownership to
Vulkan, while failed imports close the FD. Binary semaphore waits and signals
are submitted around their recorded work and retained until completion. The
Win32 route has independent-device GPU readback fixtures for buffers and images;
the FD route has a Linux cross-compilation check but no real-GPU evidence yet.

Generic device-loss state queries remain a lifecycle design target;
`DeviceLost` errors and backend-specific loss handling do not establish a
universal status/notification protocol.

### Current DX12 advanced routes

The DX12 routes below are bounded by their descriptors and runtime capability
checks. Focused real-GPU fixtures cover the cited paths; they do not establish
support for every native API option or every device.

- Acceleration structures and ray pipelines require DXR tier 1.0 and the native
  device interface. Inline ray query additionally requires DXR 1.1 and shader
  model 6.5. The backend supports the described triangle and AABB builds,
  shader groups and shader-table sections subject to its validation. The
  [ray query GPU fixture](../crates/dx12/tests/ray_query_gpu.rs) verifies hit
  and miss output; the ray pipeline example exercises a separate DXR route.
- Mesh pipelines require mesh shader tier 1 and shader model 6.5. The current
  route is restricted to its supported triangle pipeline, attachment formats,
  sample counts and depth state. See the
  [mesh GPU fixture](../crates/dx12/tests/mesh_gpu.rs).
- Sparse textures require tiled-resource support. The current route accepts
  single-layer, single-sample RGBA8 2D textures with copy and sampled usage;
  standard mips and the packed mip tail use the backend's reported tiling.
  Mapping and residency changes are exercised by the
  [sparse GPU fixtures](../crates/dx12/tests/sparse_gpu.rs).
  A sparse bind does not initialize contents or attachment compression metadata.
  Before using newly mapped RT/DS memory, callers must initialize the affected
  subresources with a complete copy or clear, or use a supported transition
  from `Undefined` on first use. DX12 lowers that first-use transition to an
  attachment-state barrier followed by a full-subresource discard. `Common`
  does not request a discard; partial remaps require an initialization strategy
  that preserves any contents the caller needs.
- The explicit multi-queue route exposes graphics and compute lanes with native
  fence waits. Timestamp queries are supported on either lane; a query set is
  bound to its first submitted lane. Occlusion and pipeline statistics are
  graphics-lane operations. Resolve still requires the caller to establish the
  relevant submission and completion evidence. See the
  [query](../crates/dx12/tests/query_gpu.rs) and
  [multi-queue](../crates/dx12/tests/multi_queue_gpu.rs) fixtures.
- GPU-count multi-draw uses `ExecuteIndirect` with a count buffer, subject to
  the argument, stride, range and state requirements in the Core contract and
  DX12 backend. The [GPU fixture](../crates/dx12/tests/multi_count_gpu.rs)
  checks the resulting output.
- The restricted standalone texture clear route above is exercised by the
  [clear GPU fixture](../crates/dx12/tests/clear_gpu.rs).

### Current Vulkan advanced routes

The Vulkan backend admits each route only after device feature, extension,
queue-family, and descriptor checks. The focused GPU fixtures in
[`crates/vulkan/tests`](../crates/vulkan/tests) and the numbered examples cover
the admitted paths on an AMD Radeon 780M; they do not imply that every Vulkan
device supports them.

- Acceleration-structure builds, ray pipelines, and inline ray query use the
  corresponding Vulkan extensions and enabled features. Ray pipeline shader
  groups and shader-binding-table regions retain their native alignment and
  lifetime requirements.
- Mesh pipelines use `VK_EXT_mesh_shader`. The selected SPIR-V mesh entry's
  output topology is checked against the public primitive topology before
  native pipeline creation.
- Sparse textures use native sparse image binds. Allocation ranges cannot be
  simultaneously bound to different tiles without the aliasing feature; the
  supported route rejects such overlap and includes sparse queue work in
  completion coverage.
- Native graphics and compute queues use timeline semaphore waits for explicit
  cross-queue dependencies. Timestamp and pipeline-statistics queries use
  native query pools and report only the fields admitted by the public layout.
- GPU-count multi-draw uses `vkCmdDrawIndirectCount` or its indexed variant.
  Standalone clears use native image clear commands for their admitted formats
  and subresource ranges.

### Metal native routes

Metal lowers operations directly to native command buffers. Its stage binding ABI
allocates independent buffer, texture and sampler namespaces per shader stage;
vertex inputs share the native buffer budget. Logical bind groups do not reserve
fixed native slots. Buffer ranges and dynamic offsets retain their bounds, and
pipeline metadata supplies the shader's alignment and runtime-length table.
WGSL uses a pinned Naga 30 Metal writer, specialized at pipeline creation.

Raster supports multiple color attachments, depth/stencil, selected device sample
counts and multiview amplification batches. Attachment clears and resolves preserve
regions outside the render area. Standalone clears, texture/buffer copies and
readback use native paths; byte-granular copies use a GPU compute fallback when
native buffer blit alignment is stricter than the public operation.

Native object/mesh pipelines use Metal source or libraries. Their output topology
must match the pipeline descriptor. Acceleration structures support triangles,
AABBs and instances. TLAS builds reserve dependencies while recording: the same
encoder may use a pending build, while another encoder must wait for successful
completion. Unsubmitted reservations roll back, and submission failure invalidates
the build. WGSL intersection queries carry private per-instance shader-table
contributions alongside acceleration-structure bindings, including descriptor
arrays and function parameters. These native details do not enter Core's API.

Converted ray pipelines are excluded from this implementation phase by user
instruction because the Metal Shader Converter SDK is unavailable. This exclusion
is not a claim that Metal lacks ray-pipeline capability. Full secondary raster
recording remains unavailable because Core requires independently pre-recorded
viewport/scissor state and Metal ICB render commands do not encode that state.
A parallel render encoder instead requires the actual parent render pass. The
backend does not replace these contracts with CPU command replay.

Graphics and compute queues synchronize through native shared events, including
cross-packet dependencies. Failed producers remain visible in consumer completion;
accepted GPU work is not rolled back. Placement sparse textures require the Metal 4
queue mapping interface and `supportsPlacementSparse`; older automatic sparse
heap allocation cannot express the public explicit allocation-offset contract.
Resource retention covers native command buffers, argument-array contents,
acceleration dependencies, external imports and sparse mapping completion.

Core's exact eleven pipeline-statistics fields differ from Metal's
[eight native statistic counters](https://developer.apple.com/documentation/metal/mtlcommoncounterset/statistic).
The separate pass-boundary statistics domain returns available raw counter fields
and invalid-value sentinels. These samples do not imply the eleven-field query or
that subtraction is meaningful. Timestamp and occlusion support are checked on
the device. GPU-count multi-draw converts the raw arguments to native ICBs in a
GPU pass before raster execution; its buffers must already be in the declared
indirect-read state before that raster pass.

Mapping uses shared host storage with exclusive mapping leases. Placement heaps,
binary archives, borrowed native interop and device-compatible external resources
remain backend-private. Surface support reports native formats, present modes and
color spaces, including device/OS-gated EDR/PQ configuration. Runtime selectors,
GPU families, format support, argument-buffer tiers and combined stage resource
limits remain authoritative.

At the user's request this phase performs compilation and static review without
Metal tests. There is no Apple GPU, generated-MSL compiler, presentation or visual
acceptance claim. See the [Metal implementation review](evidence/metal-completeness-2026-10-22/README.md).

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

Indirect draw support and nonzero first-instance support are separate facts.
`DeviceCapabilities::indirect_first_instance` guarantees both instanced attribute
addressing and the shader instance-index contribution. When it is false, callers
must write zero to that field in every indirect record, including GPU-produced
and count-controlled records. Backends do not read argument buffers on the CPU
to enforce this data precondition. Direct draws retain their ordinary range
semantics independently of this capability.

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
Callers explicitly flush modified lease-relative byte ranges before GPU readers
and invalidate completed GPU-written ranges before host readers. Lease release
does not guarantee publication of unflushed host changes. In particular, dropping
a previously flushed staging lease must not overwrite later GPU updates. GL uses
owned host storage for long-lived leases and transient native read mappings for
snapshots, so safe borrowed slices never retain a driver mapping across resets.

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

`Queries::resolve_mode()` makes native resolve ordering explicit. `Ordered`
permits resolution after the producer in submission order, without a host wait.
`AfterCompletion` requires the caller to establish producer completion first;
the backend checks native query availability before any destination write and
returns `InvalidState` if a result is still unavailable. WebGL query APIs need
this mode because they expose completed host results but no GPU query-buffer
resolve. The caller chooses completion boundaries and browser event-loop
progress. Core does not poll, delay, or replay query commands.

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
