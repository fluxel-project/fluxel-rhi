# Fluxel RHI architecture and design rationale

This document describes the current source architecture and its reasons. The
public contract lives in [`core::api`](../crates/core/src/api/mod.rs); support
queries and native implementation determine which operations a device can
execute. API vocabulary alone is not an implementation or conformance claim.

### Integration boundary for optional features

Feature implementations use Rust libraries, existing native bindings, and
native entry points that can be called directly from Rust. A feature that
requires an additional C/C++ bridge or a separate SDK solely for its integration
is treated as unsupported; such a dependency is not added to complete a backend
feature checklist. This does not remove the normal platform toolchains already
required to build a backend.

Capability reports and documentation must distinguish this integration boundary
from missing native API support or a device feature that is absent at runtime.
Loading an optional OpenGL entry point through the existing Rust WGL/EGL bindings
is within this boundary and does not introduce a C++ integration layer.

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

#### Unsupported GL-family domains

The following domains are explicitly unsupported for the GL backend family,
including desktop GL4+, GLES3+, and WebGL2. They are outside its implementation
scope, even when a particular desktop driver advertises a related extension.
Capability flags remain false, unavailable optional interfaces stay absent, and
callable unsupported operations return structured errors.

| Core domain | GL-family support decision |
| --- | --- |
| Secondary command recording | Unsupported; the backend has no equivalent native parallel command-recording model. |
| Multiple independent GPU queues | Unsupported; GL contexts do not implement Core's independent queue contract. |
| Sparse resources and placed memory | Unsupported; GL page commitment is not Core's explicit allocation, placement, mapping, and aliasing contract. |
| External memory and resources | Unsupported; no Core external-resource integration is exposed. Driver-specific EXT import routes do not enable this domain. |
| Mesh shaders | Unsupported, including vendor-specific mesh/task shader extensions. |
| Ray pipelines and acceleration structures | Unsupported. |
| Ray query | Unsupported. |

These exclusions are deliberate support boundaries, not pending GL feature work.
Completion of the GL backend is judged against its supported domains and their
documented device, format, extension, and profile limits.

#### Supported execution model

Native desktop GL and EGL/GLES contexts expose Core's scoped native interop
interface. A device guard or an exclusively borrowed encoder guard retains the
owner context and borrows a shared `glow::Context` dispatch table. Its pointer
must never be treated as a mutable Rust reference or as a WGL/EGL handle.
The guard is thread-affine; nested acquisition, reentrant RHI operations, and
acquisition during an active render, compute, or query scope are rejected.

Leaving the guard, including unwinding, restores the backend's pixel-store
baseline and captured rasterizer-discard, dither, sample-coverage, and desktop
color-logic-op enables, then invalidates its native state knowledge. Resource destruction uses
an explicit retirement entry point and a separately shared dispatch table, so
dropping an unrelated resource during the scope does not alias the provider's
mutable object tables. If native restoration fails, subsequent RHI operations
are rejected. This mechanism neither records commands nor infers hazards.

The unsafe caller coordinates native side effects, retains every referenced
resource through GPU completion, and must not delete or mutate RHI-owned objects.
Other foreign or extension-specific state needed by subsequent RHI work remains
the caller's responsibility; foreign transform-feedback and query scopes must
be closed before returning. This is context/command interop, not an API for
exporting RHI buffer or texture object names.
Browser WebGL2 does not expose a native GL dispatch-table pointer; its native
interop capability remains disabled. External memory/resource import is a
separate domain and remains unsupported for the GL family.

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

Direct GLSL sources retain their declared profile and use the existing public
clip-coordinate wrapper. Pipeline constants become named preprocessor definitions
immediately after `#version`. Constant names must be non-reserved identifiers;
values must have a finite, unambiguous spelling in that GLSL profile. Direct
modules retain immutable source and compile at pipeline materialization, after
constant insertion. WGSL overrides continue to use Naga specialization.

Desktop OpenGL SPIR-V uses the `fluxel-gl-native-spirv-v1` artifact ABI and
`opengl-spirv-1.0` environment. It requires GL 4.6 or `ARB_gl_spirv` plus the
actual binary/specialization entry points. `ShaderDesc` declares the selected
native entry, and each pipeline stage must match it. The producer targets OpenGL and the
backend's native coordinate convention; this path does not translate Vulkan
artifacts or insert WGSL coordinate fixes. Raster stages must use the same
native artifact format. A vertex-only raster pipeline receives an empty native
SPIR-V fragment stage, allowing ordinary depth-only rendering.

The direct artifact interface uses `DescriptorSet = 0`. Within each independent
native namespace (UBO, SSBO, sampled texture, storage image), `Binding` is the
flattened index of Core declarations sorted by `(group, binding, element)`;
a fixed array reserves its contiguous element span. Sampler objects follow the
texture/sampler pairing in each Core group: one sampler declaration can serve
that group's textures, or equal-sized declaration lists pair in binding order.
Paired fixed arrays use matching element indices; a single scalar sampler can
serve a texture array. Other pairings must be expressed through the WGSL route
or an artifact layout matching this native ABI.

Opaque `UniformConstant` variables require explicit `Location` decorations.
The backend reads their type, array length, binding, and location from the
artifact; it never guesses a uniform location from an active-resource index or
requires the driver to retain resource names. Array lengths in this ABI use
literal or scalar specialization constants, and interface decorations are
specified directly. Names from `OpName`, or decimal `SpecId` strings, select
typed specialization values. Each native value occupies one 32-bit word.
The parser is structural preflight, not a complete SPIR-V validator: callers
provide trusted artifacts and native specialization/linking remains authoritative.
The [ARB specification](https://registry.khronos.org/OpenGL/extensions/ARB/ARB_gl_spirv.txt)
permits reflected names to be empty. Explicit artifact locations also avoid
dependence on driver-specific uniform-location interface-query behavior.
GLES and WebGL2 do not inherit this desktop artifact path.

Native buffer operations use pointer-sized `GLintptr` and `GLsizeiptr` entry
points, including pixel-buffer offsets. The advertised addressable range is
not an allocation guarantee: actual allocation remains subject to driver memory
budgets. Browser WebGL2 retains its JavaScript API limits.

Standalone color resolve operates independently of raster passes. It preserves
texture row order and validates equal formats, matching extents, multisample
source and single-sample destination. GLES/WebGL multisample blits require
matching source/destination coordinates; offset relocation therefore uses an
immediate single-sample intermediate and a second native blit, without retaining
or replaying application commands.

#### Desktop GL4+ / WGL

The WGL provider owns the current-context relationship and borrows the native
drawable for presentation. Desktop extensions can expose capabilities beyond
GLES, but only when the required native route is available. Immediate execution
and context affinity remain unchanged by the desktop profile's larger feature
set. Native fences establish completion independently of buffer swapping.

Buffer readback encodes a GPU copy into ticket-owned staging at the requested
command position. Host access follows the caller's covering completion point;
later source writes do not change the snapshot. Query-buffer-capable contexts
resolve queries in GPU order. Older desktop contexts use `AfterCompletion`:
availability is checked without waiting and unavailable results return
`InvalidState` before any target upload, rather than blocking query retrieval.

Ordinary desktop color textures can copy into buffers without framebuffer
renderability. GL4.5 or `ARB_get_texture_sub_image` copies selected rows directly
to a PBO; older desktop contexts use `GetTexImage` into a whole-mip temporary PBO
and native buffer copies to select the requested region. Both require the actual
loaded entry point and matching transfer encoding. GPU scratch also handles
unaligned destination offsets without host readback. The older path can require
temporary storage proportional to the full mip, rather than just the region.
RGB9E5 retains its separate compatible-integer-copy route for exact packed bits.
Signed-normalized formats use the same bit-preserving mechanism with a matching
integer view-size class, preserving both encodings of negative one. Their copy
support therefore requires native image-copy compatibility and integer readback;
numeric pixel conversion alone does not establish this capability.

Depth16, Depth32Float, and standalone Stencil8 support CPU and PBO uploads and
buffer readback where the format and framebuffer facts permit. For combined
depth/stencil textures, stencil transfers use one byte per texel and D32S8 depth
uses a 32-bit float. D24 depth has no portable byte representation. Selected-plane
texture copies use framebuffer blits and preserve the other plane. Uploads use
a same-format scratch texture followed by a selected-plane blit; PBO stencil
packing additionally requires GL4.3 compute/SSBO support. Temporary GPU work is
issued immediately and does not retain an application command stream.

Desktop `OVR_multiview2` uses the same contiguous-view contract as GLES. Nonzero
attachment bases are represented with native texture-view aliases, keeping the
OVR attachment base zero and retaining the aliases through pass teardown. Those
ranges therefore require a creatable texture view. There is no native arbitrary
view-mask parameter in this route. Pass-boundary timestamp writes are issued
before binding the multiview framebuffer and after unbinding it, so they remain
outside the restricted framebuffer interval. Timestamp writes inside an active
multiview pass remain unsupported: the [OVR specification](https://registry.khronos.org/OpenGL/extensions/OVR/OVR_multiview.txt)
leaves their results undefined without EXT_multiview_timer_query. Browser
boundaries likewise use instantaneous queryCounterEXT, without an active
elapsed-time query spanning multiview draw or clear commands. Compressed color
clears also remain outside
the native route: [ARB_clear_texture](https://registry.khronos.org/OpenGL/extensions/ARB/ARB_clear_texture.txt)
explicitly rejects compressed internal formats, which cannot be framebuffer
color attachments. This backend does not add a software block codec.

Conservative rasterization selects the native extension flavor. NV's route
retains its point/line support; Intel's independent token is admitted only for
filled triangle lists or strips. Extension-selection and topology checks have
unit coverage; Intel-extension GPU execution remains unverified on this host.

#### GLES / EGL

EGL owns a different context/drawable boundary while reusing GL-family lowering
and state semantics. GLES limits and extensions are queried for the actual
context, including Android contexts. This avoids both a lowest-common-
denominator public API and accidental inheritance of desktop-only features.
Android window and foreground/background events belong to platform integration;
the backend handles the resulting drawable and device operations.

GLES3+ uses bounded runtime-array specialization; nonuniform sampled-texture and
storage-buffer indexing additionally requires storage buffers. Counts still
consume native resource slots and generated sampler pairs. Query resolve uses
`AfterCompletion` and rejects unavailable slots before modifying the destination.
Buffer readback snapshots the requested command position in GPU staging.

ES3.1 compute and sampled texture support provide depth/stencil buffer copies:
depth is fetched into a temporary GPU buffer as D16 UNORM bits or D32 float bits,
and stencil as unsigned bytes, then copied into the requested padded layout.
The implementation restores native sampling, program and buffer bindings.
D24 depth retains its opaque representation. Independent stencil textures
require ES3.2 or the acquired `OES_texture_stencil8` extension. Depth uploads use
same-format 2D scratch images and masked framebuffer blits, preserving the
unselected plane of combined formats. CPU and PBO paths retain distinct source
rows, mip/layer selection and byte padding.

Ordinary color buffer copies require either an exact native read-format/type
probe or compatible raw image copies through integer textures. ES3.2 or acquired
copy-image extensions, format compatibility and integer readback evidence gate
the latter route. Numeric conversion is not used to claim preservation of
SNORM's two negative endpoints or floating-point payload bits.
Compressed block readback additionally requires a successful per-format raw
block-copy probe. A copy-image extension and integer readback support alone do
not establish compressed-to-integer compatibility on the actual runtime.
Failure of this raw route does not disable ordinary texture copies.

`EXT_buffer_storage` on ES3.1 provides persistent coherent mapping when the
extension-specific entry point is present. Readback staging copies from its
existing mapping after completion, rather than mapping the same store twice.
Contexts without that extension issue a private GPU staging copy and fence for
non-persistent map requests. The owner-thread future reuses the completion wait
primitive and reads staging only after the fence signals. Cancellation releases
the pending map lease; map usage and device/range checks apply before issuing
native work. Snapshot write leases publish only flushed subranges, and explicit
invalidation still requires the caller to establish producer completion. Local
GPU checks cover an ANGLE D3D11 runtime and a separate installed ANGLE Vulkan runtime;
an extension absent on one is not classified as absent throughout GLES.

GLES3.1 compute, SSBO and indirect draw support also provide GPU-count drawing.
At the call position, native buffer copies snapshot the count and compact all
argument records before any draw. A bounded compute utility masks records above
the GPU count and rebases indexed offsets; native indirect draws then consume
the resulting records. SSBO processing is chunked by actual device limits, and
the raster program and buffer bindings are restored. This uses O(maximum draw
count) temporary bytes and native draw calls without CPU count readback or an
application command cache. Desktop GL continues to prefer its native count
entry point. Ordinary indirect ABI and device restrictions still apply.

#### Browser WebGL2

WebGL2 uses browser-thread JavaScript objects and the immediate contract. The
browser implementation includes a typed GL-family provider and a Core
`BrowserOwner`/`BrowserEncoder` execution path; they have distinct native call
boundaries and need their own validation. Neither is a deferred CPU command log.

Runtime-sized sampled-texture and sampler arrays specialize a declared finite
maximum into ordinary native slots. Native slot limits are checked before shader
expansion; partial bindings and short prefixes retain the same Core rules as
fixed arrays. This does not expose an unbounded descriptor heap. The combined
nonuniform texture/storage-buffer capability stays disabled because WebGL2 has
no storage-buffer interface.

Mapping issues a GPU copy into a private staging buffer and inserts a fence.
Its future yields to the browser until that fence signals, then obtains the CPU
snapshot with `getBufferSubData`. Cancelling the future releases the pending map
lease and retains native completion safety. Explicit invalidation refreshes a
snapshot only after the caller has established completion for the source work.
There is no blocking fence wait or CPU snapshot during the initial map request.

Direct ordinary integer and UNORM texture-to-buffer transfers require a successful
byte-comparison probe for the actual attachment and readback encoding. R8/RG8
UNORM use portable RGBA8 pixel-pack reads followed by GPU buffer subrange copies
that retain only the declared channels. This fallback issues one read per row
and one native copy per texel; it never downloads intermediate pixels to CPU.
An available texture-to-texture copy does not imply exact raw-buffer readback.
SNORM duplicate endpoint encodings and floating-point NaN payloads cannot be
promised through numeric `readPixels` conversions.

When vertex `highp` exposes at least 23 mantissa bits, normalized transform
feedback reconstructs R16/RG16/RGBA16 UNORM and RGB10A2 into their exact integer
encodings even when direct pixel-pack pairs are unavailable. It supports 2D,
array, and 3D color regions with independent row/image strides. This fallback
does not admit non-injective SNORM or floating-point bit representations.

Single-sample 2D Depth16 supports CPU uploads, GPU pixel-unpack uploads, depth
copies, and exact buffer readback through a private transform-feedback draw;
the readback also supports depth-array subregions with independent image stride.
The draw reconstructs two 16-bit normalized values per output word, followed by
GPU copies into the caller's padded rows. Private bindings and objects are
scoped to the operation. Depth PBO uploads compact the requested source rows on
GPU and upload the complete rectangle in one call; no CPU readback is involved.
Depth32Float CPU and PBO uploads are validated by numeric depth sampling, which
does not establish a portable raw floating-point depth readback representation.

Stencil8 renderbuffers accept attachment and copy usage, including copy-only
resources, while remaining unavailable for sampled/storage bindings. Stencil
byte-plane uploads use a private R8UI texture and eight fixed stencil-write
draws. Readback uses stencil tests to produce exact Boolean color channels,
then transform feedback packs those channels into bytes. The same operations
support the stencil plane of packed depth/stencil textures, including array
subregions, without relying on stencil texture sampling. They reject active
query, transform-feedback, or raster scopes rather than altering their results.

Depth32FloatStencil8 depth-plane uploads expand compact float texels into a
private packed PBO, use the native packed PBO upload overload, and blit only
depth into the destination. The stencil plane remains unchanged. Both upload
directions have output tests that observe preservation of the other plane and
subsequent ordinary raster work. Depth24Plus remains opaque: there is no
portable raw 24-bit depth byte plane to expose.

The following remaining limits follow the browser API rather than missing Rust
bindings. They are reported as unsupported instead of introducing CPU execution
or a second scheduler:

| Domain | WebGL2 boundary |
| --- | --- |
| Compressed texture clear | No native clear of arbitrary compressed blocks and no renderable compressed attachment; software codecs are outside this backend. |
| Arbitrary multiview mask | OVR exposes one contiguous base/count range, with a common base for framebuffer attachments. |
| Multiview with an active timer query | OVR explicitly rejects draw/clear with more than one view while a timer query is active. |
| GPU-count multi-draw | `WEBGL_multi_draw` consumes CPU arrays, with no indirect argument/count-buffer execution interface. |
| Pipeline statistics | Available query targets do not express Core's complete pipeline-statistics fields. |
| Serialized native program binaries | WebGL2 removes the program-binary entry points. Shader-source reuse does not constitute a native binary cache. |
| Polygon point mode and conservative rasterization | No matching exposed browser route. Line polygon mode, depth clamp, and other optional raster states use their actual browser extensions when available. |

These limits are defined by the [WebGL2 specification](https://registry.khronos.org/webgl/specs/latest/2.0/),
[OVR multiview extension](https://registry.khronos.org/webgl/extensions/OVR_multiview2/),
and [multi-draw extension](https://registry.khronos.org/webgl/extensions/WEBGL_multi_draw/).

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
render-attachment usage and a graphics lane. It supports device-renderable float
and integer color formats and depth clears, including selected mips and layers.
The depth clear preserves stencil; Core does not define a standalone stencil
clear value. Render-pass attachment load clears are a separate operation.
DX12 standalone buffer clear writes a repeated little-endian `u32` to a nonempty
four-byte-aligned `COPY_DST` range in device or readback memory. A single bounded
upload pattern is reused by native copies; the destination remains `COPY_DEST`.

DX12 standalone color resolve accepts matching multisample/single-sample 2D
formats with native `MULTISAMPLE_RESOLVE` support. Source usage includes
`RENDER_ATTACHMENT` or `COPY_SRC`; destination usage includes
`RENDER_ATTACHMENT` or `COPY_DST`. The caller establishes `ResolveSrc` and
`ResolveDst` states. Whole subresources use `ResolveSubresource`; partial
rectangles with independent offsets and array layers use command-list version
1. This average-color operation does not define a depth/stencil resolve policy.

DX12 placed memory uses native allocation requirements and retains the heap
inside every placed buffer or texture. Native memory classes are default mixed
heaps (0, resource heap tier 2 only), upload buffers (1), readback buffers (2),
default buffers (3), non-render-target/non-depth textures (4), and render-target
or depth textures (5). Select a class from `native_type_bits`; placement validates
the exact device, class, range and alignment. Mixed and RT/DS heaps use 4 MiB
heap alignment to accommodate multisampling. Heap allocation policy and alias
selection remain with the caller. Explicit Core alias barriers lower to native
aliasing barriers and retain both resource owners through GPU completion.
They do not transfer initialized contents into the newly activated native
resource identity. A placed attachment first used through
`Undefined -> ColorAttachment` or `Undefined -> DepthStencilAttachment` lowers
to a whole-subresource `DiscardResource` before the application initializes it
with a full clear, copy, or render pass. The same lowering applies to sparse
attachments; it is a native execution detail, not resource-state inference.

Resource-less global barriers preserve resource states and texture layouts.
DX12 uses enhanced global barriers when options 12 and command-list version 7
are available, with queue-compatible synchronization and conservative cache
scopes where Core does not distinguish native accesses. `ALL` covers the GPU
work supported by that queue, including ray dispatch and acceleration builds
on compute queues. Older runtimes retain a restricted null-UAV barrier route;
general copy/attachment scopes require enhanced barriers. Explicit host scopes
need completion and mapping synchronization, not a command-list memory barrier.

DX12 scoped native interop returns a borrowed `ID3D12Device` or the actual
`ID3D12GraphicsCommandList` COM pointer. The guard owns an interface reference;
encoder access also holds Core's exclusive encoder borrow. Foreign commands
must restore native state, obey explicit resource states and retain their own
dependencies through GPU completion. They must not close, reset or separately
submit the borrowed list. An external acquire/release may open a new native
list, so foreign code must reestablish its own bindings after such a boundary.

External resource import is an explicit bridge contract. The bridge provides
an owned native handle and the producer's exact allocation and resource
metadata, and guarantees matching ownership transfers and synchronization.
An imported buffer or texture can be obtained for ordinary commands and
bindings. Texture transfers specify the subresources, usages and old/new
layouts. The producer's synchronization payload is imported before acquire;
release returns a payload signaled only when the recorded work is submitted
and reaches the release. The bridge must not use the resource outside the
matching ownership interval or wait twice on a binary semaphore payload.

DX12 imports owned NT resource handles and shared-fence handles with explicit
timeline values. Opened COM resources outlive the consumed handles. Texture
metadata, buffer memory class, native flags and binding capacity are validated
against the imported resource; an object handle is not an implicit buffer slice.
At each explicit acquire/release, the backend closes the current native list
and opens the next. Submission executes those lists and queue fence waits or
signals in exactly the recorded order. Ordinary work without such boundaries
retains batched list submission. These native segments retain COM objects and
allocator ownership through fence retirement, including indeterminate failure;
they store no CPU draw/dispatch replay stream and derive no dependencies.

Vulkan currently imports opaque Win32 handles or opaque file descriptors when
the relevant extensions and per-resource import features are available. Win32
handles are closed after import; successful FD imports transfer ownership to
Vulkan, while failed imports close the FD. Each explicit external acquire or
release ends the current native command-buffer segment. Binary waits attach to
the following segment and binary signals to the preceding segment, preserving
commands before acquire and after release. Plan timeline waits and completion
signals still wrap the first and final submissions. Ordinary unsegmented work
retains batching; every segment, semaphore and command pool remains alive until
completion or the existing indeterminate-handoff quarantine. No draw/dispatch
commands are saved for CPU replay. The
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
- Sparse textures require tiled-resource and per-format support. The route
  supports validated single-sample 2D arrays and tier-3 3D resources, including
  native-supported color, BC and depth uses. Standard mips and packed tails use
  the returned native tiling. Array packed tails require tier 4 and consistent
  per-slice metadata; the current test device lacks this optional capability,
  so its tier-4 fixture reports a runtime skip rather than GPU acceptance.
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
- Standalone texture clear uses the attachment-state contract described above;
  it cannot run on a compute queue merely because D3D12 also offers a different
  UAV-state clear operation. Standalone buffer clear and partial color resolve
  have focused [buffer-clear](../crates/dx12/tests/clear_buffer_gpu.rs) and
  [resolve](../crates/dx12/tests/resolve_gpu.rs) readback fixtures.
- Bounded runtime WGSL binding arrays specialize to the selected pipeline
  layout's `RuntimeSized` capacity; an equally sized fixed layout is not
  interchangeable. The backend-specific `subgroups` shader requirement needs
  native WaveOps. Required compute subgroup size additionally needs shader
  model 6.6 and a supported native wave size; DXC compiles the selected WGSL
  entry with `WaveSize`. Direct HLSL sources use `Hlsl(major * 10 + minor)`:
  shader models 60–69 are checked against the device before admission. The
  selected vertex, fragment or compute entry is compiled at PSO creation with
  DXC, using the same native binding ABI as DXIL: Core group maps to register
  space and binding maps to its register within the CBV/SRV/UAV/sampler class.
  Vertex-buffer locations use `LOCATION<n>` input semantics; Naga-generated
  WGSL uses its separate `LOC<n>` convention.
  Overlapping ranges within a native register class are rejected; WGSL uses
  its separate layout-aware remapping. Native HLSL coordinates and builtins
  receive no WGSL coordinate or vertex-index rewrite.
  HLSL pipeline constants are validated identifier/finite-value `-D` macros;
  integer widths, finite half values and full-precision double values retain
  their numeric meanings. A requested wave size first preprocesses macros and
  conditionals in memory, then annotates the active `void entry(...)` definition.
  Combining an existing `WaveSize` annotation with the pipeline option is
  rejected. Source includes
  require provenance and an include resolver absent from the bytes-only Core
  input, so this route does not resolve filesystem includes.
  Direct DXIL artifacts cannot be modified by this
  pipeline option or by post-hoc specialization constants.
- Native pipeline statistics are interval counters. They do not implement
  Core's separate Metal-style vertex/fragment start/end sample domain.

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
- Sparse capability may come from either created graphics or compute lane.
  Submission checks that the selected lane itself supports sparse binding;
  an uncreated queue family cannot supply a reported capability. A separate
  sparse-only queue has no lane role in the current Core model.
- Standalone color resolve preserves caller-selected `ResolveSrc`/`ResolveDst`
  states, which map to `GENERAL` on Vulkan. Matching multisample/single-sample
  2D regions use `vkCmdResolveImage` directly, accepting render-attachment or
  corresponding copy usage. Dynamic-rendering resolve targets use the same
  `GENERAL` layout. Resolve barriers conservatively cover both transfer and
  attachment access without inferring state from creation flags. Native image
  allocation retains the transfer usages needed by the admitted resolve paths;
  full format/usage/extent/sample combinations are queried before admission.
  Vulkan requires a graphics-capable command-buffer family for this command.
- Shader subgroup admission checks supported stages and operation families.
  Direct SPIR-V capabilities are checked against those native facts; quad
  operations remain valid in fragment and compute stages when
  `subgroupQuadOperationsInAllStages` is false. WGSL subgroup builtins use the
  backend-specific `subgroups` requirement; the pinned Naga parser does not
  accept the `enable subgroups;` directive. Subgroup-size control remains a
  separate device property and compute pipeline option.
- Supported 8/16-bit shader and storage features are explicitly enabled on the
  logical device using coherent Vulkan 1.1/1.2 feature nodes. Direct SPIR-V
  precision/storage capabilities are checked against enabled features, including
  the base 16/64-bit integer and 64-bit float features. Native specialization
  retains each Core scalar representation. The
  [precision fixture](../crates/vulkan/tests/assets/precision/precision.comp)
  and [GPU check](../crates/vulkan/tests/subgroup_gpu.rs) cover half/int8
  specialization and 8/16-bit storage readback without a runtime SDK compiler.
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
the device. GPU-count multi-draw prepares native ICBs at the count-draw command
position, suspending and reopening the render encoder only when needed. Its
buffers must already be in the declared indirect-read state before that raster
pass.

Mapping uses shared host storage with exclusive mapping leases. Placement heaps,
binary archives, borrowed native interop and device-compatible external resources
remain backend-private. Surface support reports native formats, present modes and
color spaces, including device/OS-gated EDR/PQ configuration. Runtime selectors,
GPU families, format support, argument-buffer tiers and combined stage resource
limits remain authoritative.

At the user's request this phase performs compilation and static review without
Metal tests. There is no Apple GPU, generated-MSL compiler, presentation or visual
acceptance claim. The implementation review found no Apple GPU execution
evidence; the published capability status therefore remains limited to the
checks available on this host.

### GL recording review validation

The focused recording review preserves immediate native execution and the GL
state cache. It adds no command scheduler, resource-state history, or draw replay.
Readback tickets capture their command position in GPU staging, including when
the source is overwritten before completion. Non-query-buffer resolve uses
`AfterCompletion` and checks availability before reading results or modifying
the destination.

Focused hardware checks on Radeon 780M cover WGL persistent mapping, ANGLE
D3D11 GLES non-persistent staging mapping, cancellation/remapping, explicit
flush/invalidation, readback snapshot preservation, and GLES occlusion resolve.
WGL also validates the non-query-buffer timestamp route and multiview pass
boundary timestamp results with surrounding sentinel bytes. Chrome WebGL2
checks mapping and completion-gated occlusion. The tested GLES and browser
runtimes do not expose usable timestamp counters; their multiview timestamp
checks skip that optional combination and do not establish positive coverage.
These checks do not establish new Surface, visible-example, or all-format
acceptance. Metal retains the separate no-runtime-tests boundary above.

### Binding bounds are caller intent, validated against native limits

`BindingCount::RuntimeSized { max_count }` supplies a nonzero layout capacity.
A bind group supplies a nonempty prefix within that bound; shaders may access
only supplied elements, and partially bound holes require explicit support.
Core and backend bind-group creation validate the same bound; pipeline creation
also validates the native resource slots consumed by specialization. Capacity belongs to
the caller's workload, rather than being inferred from a driver maximum or
turned into a Core allocation policy.

Desktop GL4+ and GLES3+ specialize runtime binding arrays to this layout capacity before
GLSL generation, using the ordinary fixed-array binding path. Nonuniform
sampled-texture and storage-buffer indexing is available on native contexts
with storage-buffer support. Short prefixes and explicitly permitted holes
use typed dummy slots; accessing an absent element remains invalid caller
behavior. UBOs, SSBOs, sampled textures, samplers, and storage images retain
their individual resource support checks. Specialization uses native slots,
not bindless handles: per-stage limits and the generated texture/sampler pair
count bound pipeline creation. Divergent implicit samples evaluate candidates
before selection to preserve derivatives, which can increase sampling cost.
WebGL capability declarations remain separate from native contexts.

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
a previously flushed staging lease must not overwrite later GPU updates.
Desktop GL4.4/`ARB_buffer_storage` and GLES3.1/`EXT_buffer_storage` use immutable, persistently mapped coherent
storage for mappable buffers. A lease retains the buffer and its context until
CPU access ends; native unmapping and deletion occur only during retirement.
Mapping an existing allocation introduces no GPU wait or copy. Completion and
host/GPU exclusion remain caller obligations; coherent flush/invalidate need
no additional driver operation. Coherent writes may become visible before
flush, and invalidation may expose completed writes outside the requested
range. Older native profiles and WebGL retain owned CPU snapshots and can
require synchronous refresh. Graphics reset recovery requires recreating the
context and objects; ordinary mapping tests do not establish recovery behavior
for outstanding leases.

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

## Example execution acceptance

The native example acceptance policy is separate from compile availability.
An unsupported native entry point exits with a failure status. WASM module
initialization is not GPU execution: browser checks must await the applicable
canvas entry point and inspect its GPU/output result.

## Core and examples synchronization review

Exclusive Vulkan resources crossing native queue families require matching
`OwnershipTransfer::Release` and `Acquire` barriers. A semaphore wait alone does
not transfer resource ownership. Example 23 supplies both compute-to-graphics
halves and returns each rendered particle buffer to compute before its next
use. Its graphics wait includes COPY and VERTEX because the first frame copies
particle data before vertex consumption. The existing host completion boundary
orders the next frame; RHI does not infer a state history or schedule transfers.

Examples 19, 21 and 23 preserve the original error for rejected, partially
accepted and indeterminate submissions. Potentially issued outcomes stay alive
through the existing GPU drain. A failed drain reports both failures and stops
without unwinding; it is not treated as completion or successful recovery.

Core's additional integrity checks do not change execution models. An oversized
`SubmissionPlan::step` index produces an invalid sentinel instead of wrapping
onto a valid step; validated plans cannot contain that sentinel index. Encoder
creation checks a published lane inventory before SPI. Same-plan GPU signal
consumers are checked against their descriptor index and lane; future-plan
consumers remain valid and are checked when their actual wait is submitted.

Focused validation covers 64 Core tests, an explicitly required Graphics family
0 / Compute family 1 three-round exclusive-buffer transfer/readback scenario,
Vulkan multi-queue tests under Khronos synchronization validation, and four
frames of Example 23 on both Vulkan and DX12 with particle/image readback.
All 23 unsupported Linux native entries exit with status 1; browser example
compilation is recorded separately. This is focused acceptance, not a new
all-platform or Metal runtime validation claim.

## DX12 native-object and dependency review

DXGI enumeration exposes every native adapter ordinal, and device opening uses
the selected ordinal with a checked conversion and an actual D3D12 device
creation check. The default helper still selects the first enumerated adapter;
explicit adapter choice belongs to the caller. Texture support reports the base
format and declared compatible view formats admitted by the same native checks
used for view creation. Neither change expands the set of DXGI texture formats.

Compute blits reuse device-owned immutable root signatures and PSOs keyed by the
exact shipped shader variant. Descriptors and scratch resources remain per-work
retention. The cache has a finite set of 36 typed-UAV and 12 buffer variants;
no application commands or execution schedule are cached. Enhanced-barrier
support is an immutable OPTIONS12 fact queried once during device opening.

The WGSL vertex-index indirect ABI caches its indexed/non-indexed command
signatures on the matching raster pipeline, whose root signature they reference.
A device-cached compute kernel packs larger batches directly on the GPU. Its
root signature and PSO are prepared during cold WGSL pipeline creation. A
measured one-to-three-record native-copy fast path bounds copying to six calls,
avoiding fixed dispatch/descriptor overhead for those tiny batches. Larger
record counts do not generate a CPU-side per-record copy loop.

TLAS dependency retention separates the latest accepted build from recorded or
in-flight work. Recording/cancelling a build does not replace committed BLAS
dependencies. Before native execution, the submitted batch retains the union of
current dependencies and new build inputs, including primary and secondary
bindings that can perform inline ray queries. Only builds in an executed native
command-list segment publish new committed dependencies. The batch's existing
fence retirement or indeterminate-work quarantine retains old dependencies; the
TLAS no longer permanently appends every historical BLAS. This metadata owns
native lifetimes and does not infer hazards, replay commands, or schedule work.

Focused native checks cover DXGI adapter LUID identity, declared view creation
and sRGB sampling, repeated immutable blit-object reuse, repeated blits followed
by user compute work, and 32 TLAS rebuilds with cancellation and actual old-owner
release checks. Ray-query and ray-pipeline output checks remain separate gates.
The serialized DXR pipeline-cache limitation remains explicit and nonblocking:
the existing [pipeline library](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nn-d3d12-id3d12pipelinelibrary)
caches PSOs, while DXR uses [state objects](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nn-d3d12-id3d12stateobject).
Supplying a ray cache is rejected rather than silently treated as a cache hit.

## GL-family multisample copy boundary

Desktop GL 4.3+ or `ARB_copy_image` may copy matching-sample color texture
images directly through `CopyImageSubData`. Admission requires the loaded
native entry and exact probed format, sample count and texture kind; ordinary
MSAA textures and multisample arrays have separate facts. This permission does
not enable multisample pixel uploads, buffer readback, or a framebuffer-copy
fallback. Resolve remains a separate operation that reduces sample count.
Selected depth/stencil-plane copies remain on their existing restricted paths.

GLES same-sample MSAA texture copy is **Unsupported in this backend**, including
ES 3.2 core and ES 3.1 `EXT_copy_image` / `OES_copy_image` contexts. Discovery
keeps multisample copy-source/destination facts false. A multisample color
attachment may still carry `COPY_SRC` for the separate standalone resolve
operation; `COPY_DST` does not acquire same-sample MSAA copy support. Ordinary
single-sample GLES image copies and MSAA-to-single-sample resolve are unchanged.

This is an explicit project validation restriction, not a claim that the GLES
standard lacks image-copy capability. Software Mesa tests passed, but WSL
Mesa/D3D12 produced invalid source readback and native crashes, including a
sample-mask/resolve control that crashed after correct readback. The additional
LDPlayer Android 14 GLES 3.1 test passed the 2D 4x sample-mask/copy/resolve workload
and normal context teardown. Its multisample-array storage allocation and sample
query succeeded, but attaching layer 1 returned
`FRAMEBUFFER_INCOMPLETE_ATTACHMENT`; RHI therefore rejected that array route
before copying. The emulator reports an Adreno renderer through its translation
layer; this does not establish execution on a physical Adreno GPU.

Those observations neither resolve the earlier native failures nor complete
GLES array-copy acceptance. At the user's request, the backend disables the
whole GLES same-sample MSAA-copy feature instead of advertising the partially
verified route. Re-enabling requires focused per-sample 2D and nonzero-array-layer
copy/resolve/readback checks with clean native execution and teardown on a
usable GLES runtime, plus resolution of the outstanding failure classification.
No speculative driver blacklist, EGL teardown workaround or CPU copy is added.

WebGL2 cannot use this route: its API exposes no CopyImage entry point, and its
ES 3.0 framebuffer-blit rules reject multisampled draw framebuffers. A native
MSAA-to-single-sample resolve does not establish same-sample image-copy support.

## Metal recording and synchronization review

An ordinary raster pass stays in its encoder's original `MTLCommandBuffer`.
Only a nonzero GPU-count draw suspends the native render encoder, retains the
attachments with Store, generates its ICB in a compute encoder, and resumes with
Load in the same command buffer. Native pipeline/binding/dynamic state is
reinstalled; previously encoded draws are never replayed. Final attachment
store/resolve actions and timestamp ends remain at the logical pass boundary.
Occlusion segments use private visibility buffers and a device-cached native
merge pipeline to preserve the portable zero/nonzero result. Pass-statistics
segments retain the first start sample and serialize end-sample writes; ordinary
passes without statistics incur none of that counter synchronization.

Every transition batch is range/aspect validated before native encoding. Ordinary
resource-specific transitions use native tracking only when the actual Metal
resource reports Tracked. Untracked resources, global boundaries and aliasing
retain full event synchronization; one event is allocated lazily per encoder
and its monotonically increasing values are reused. A transition cannot insert
a command-buffer event inside an active render/compute encoder. Heap-default
untracked resources are not mistaken for tracked device allocations. This
lowers explicit dependencies and introduces no resource-history tracker.

`nextDrawable` is synchronous even though the Surface SPI wraps its result in
`Ready`. Drawable exhaustion can block the calling thread until availability or
the configured timeout. The host/renderer must bound frames in flight and
acquire late; RHI does not add a drawable-acquisition thread pool.

Shared staging uploads now suballocate encoder-local slices from device-owned
native MTLBuffer pages. Submitted slices are retained by the native completion
handler before commit; cancellation releases only unsubmitted leases. Idle pages
hold native buffers without a Rust DeviceState reference and are cleared during
device teardown. This removes per-write native buffer creation; no frame-time
improvement or Apple GPU acceptance is claimed without measurements. Metal tests
remain omitted at the user's request.

## Metal transfer lowering

`clear_buffer` preserves Core's repeated 32-bit word semantics. Words made of
one repeated byte use Metal's native byte-fill operation; other values use a
bounded 64 KiB shared upload pattern and native buffer copies. The command
buffer retains the upload through GPU execution. The operation requires an
idle encoder, `COPY_DST`, and a nonempty in-range whole-word interval.

Color blits for single-sample 2D textures and array slices use native sampling
and render commands between private scratch textures, followed by an exact
copy into the destination region. This preserves the public `COPY_SRC` /
`COPY_DST` contract even for imported textures without shader-read or render-
target usage. It supports mip levels, offsets, scaling, same-numeric-class
format conversion, and nonoverlapping regions of the same texture. Integer
sources use nearest reads; linear filtering checks native format/family facts.
Framebuffer-only resources and unsupported shape/format combinations return
structured errors before lowering. This is direct native recording, with no
CPU replay or resource scheduler.

Standalone color resolves accept render-attachment or copy-source usage on
the source, and render-attachment or copy-destination usage on the target. Their
private native attachments carry the additional render-target permissions.
These additions received Apple-target type checking and independent static
review only; Metal tests and GPU acceptance were intentionally not run in this
phase at the user's request.

## Browser WebGPU native operation coverage

Standalone color resolve records an empty native render pass with `loadOp:
load`, `storeOp: store`, and a resolve target. A full, equally sized attachment
pair resolves directly. Subrectangles and different destination sizes use a
full-source-mip, single-sample scratch texture; an exact native copy writes a
`COPY_DST` target, or a scissored `textureLoad` draw writes an attachment-only
target. The source and pixels outside the destination region remain defined.
This path requires the native WebGPU multisample shape and renderable color
resolve format. The standard format table excludes integer, 32-bit float, and 16-bit normalized
color resolves even where multisampled attachments are available; these are
rejected before recording. Depth/stencil resolves have no equivalent standard
interface.
No CPU sample processing or command replay is involved.

Standard indirect draws are available independently of the optional
`indirect-first-instance` feature. The latter controls only nonzero first
instance values. Multi-draw uses direct native encoding, with GPU-generated
arguments for count-buffer variants. Core `StorageBufferView` bindings lower
to native buffer, offset and size after Core validates their view contract;
there is no separate WebGPU structured-buffer-view object. Native alignment
and resource limits still apply. The backend requests `core-features-and-limits`
when the adapter advertises it, alongside the existing optional native features.

## Verification and extension boundary

The [testing guide](testing.md), [conformance matrix](conformance-matrix.md),
and [example instructions](../crates/rhi/examples/README.md) record verification
scope separately from architecture. Android verification covered the tested
Vulkan, GLES, and WebGL2 example routes; WebGPU adapter acquisition was
unavailable in that environment. It does not imply universal Android driver
support or close renderer-owned restoration integration.

Compilation, successful submission, and native validation alone do not prove
correct GPU output or presentation. A new route needs consistent public support
answers, native creation/execution/lifetime behavior, focused real-GPU output
checks, and surface/visual evidence where applicable. Extending a backend must
preserve these ownership boundaries; a consumer fallback or another backend's
successful run cannot stand in for its missing native implementation.

## Vulkan native lifetime and allocation review

TLAS storage retains the latest accepted BLAS dependency set rather than every
historical rebuild. Recording and cancellation leave this set unchanged. Before
native submission, a batch pins both current dependencies of bound TLAS objects
(including secondary bindings) and the new build inputs. Each command-buffer
segment publishes its new dependency set only after successful native submission.
Existing fence retirement or uncertain-submission quarantine retains the old
owners; this metadata does not plan hazards or replay GPU commands.

Buffer and texture writes suballocate persistently mapped, host-coherent staging
pages. Pages are local to a queue family because their buffers use exclusive
sharing. Encoder leases keep copied bytes alive until cancellation or submission
fence retirement, and return pages before releasing their device owner. Idle
native pages do not retain DeviceState, avoiding a device ownership cycle.
Texture layout offsets and native copy alignment both apply to suballocations.

The completion observer blocks on job arrival while idle. Active fences are
grouped by logical device for zero-time wait-any checks; per-fence status calls
are only needed when that device batch reports progress. Observation backs off
from 250 microseconds to at most 2 milliseconds, with bounded intake so a
producer burst cannot starve retirement. Future polling only installs a
replaceable, cancellation-safe wake registration. Independent devices can
publish while another fence remains blocked; failed native observations retain
uncertain owners. Optional capture has a separate bounded worker and cannot
block completion observation. Saturated diagnostic capture may be omitted.

Command pools use exclusive queue-family/command-level leases and reuse reset
command buffers. Secondary recording keeps its native inheritance and simultaneous
use semantics. Bind groups and sampler variants lease single-set descriptor
pool pages. Pools reset only after their owners retire; inactive command and
descriptor caches each retain at most 32 pages, destroying excess pages without
changing public capability. These caches contain native allocation objects,
not application commands, pass schedules, or a second RenderGraph.

A local native-object microbenchmark on Radeon 780M measured median command
pool create/allocate/begin/end/destroy at 212.253 microseconds versus 1.630 for
reset/rebegin; descriptor create/allocate/destroy was 1.137 versus 0.030 for
allocate/reset. These diagnostic object-operation measurements motivated reuse;
they are not renderer frame-rate claims.

WSI acquisition retains the worker-side acquire-fence handshake and cancellation
model. A waker-driven, two-window diagnostic measured 60 warmed acquisitions per
mode: median 563.8 microseconds for FIFO and 704.4 for Immediate, with respective
95th percentiles 2811.6 and 2375.5 microseconds. These end-to-end measurements do
not isolate the fence's cost or establish a bottleneck. Replacing the handshake
with GPU-only semaphore acquisition remains deferred until a measured candidate
justifies the additional cancellation and present-retirement machinery. Retired
swapchains are released after a replacement image has presented and its present
wait has been proven consumed on reacquisition, not on the first acquire alone.

Focused acceptance includes 22 library tests (including the real-device TLAS
rebuild/cancellation and independent-fence observation regressions), three
explicit hardware fixtures for staging reuse/offset readback and distinct queue
families, and native mapping, copy, sampler, MRT and WSI tests. The distinct-family
fixture exercises Graphics family 0 and Compute family 1. Example 13 also runs
12 frames of native parallel secondary recording; its GPU frame readback was
opened and checked for the expected UFO field and star background.

Strict synchronization validation additionally exposed a surface fallback-resolve
bug: a nonzero-layer MSAA source used a transfer resolve into an attachment-layout
surface without the matching layout and transfer-write dependencies. The backend
now brackets that resolve with color-attachment/general transitions and restores
the surface's exit state. Both MRT/array resolve and WSI presentation regressions
pass without validation errors after the fix. The independent source review
confirmed no further P0/P1/P2 findings in the reviewed paths; this remains focused
Windows/780M acceptance, not an all-device or Metal test claim.


## Metal review dispositions

Adapter discovery uses the pinned objc2-metal 0.3.2 platform wrapper. Its
`device.rs` implements MTLCopyAllDevices through the native enumeration API on
macOS/Catalyst, and through MTLCreateSystemDefaultDevice on other Apple platforms.
The review's unavailable-iOS-symbol diagnosis therefore does not apply to this
call site; the Rust wrapper is not a direct call to that symbol on iOS.

Indirect-count conversion keeps its immutable Metal library, function and compute
pipeline in the device-owned converter. Each recording gets a fresh argument
encoder because its bound argument-buffer state is mutable. Capability probing
and actual execution use the same successfully created converter. This caches
native compilation objects without caching or replaying application commands.

Mesh output topology is part of the native shader ABI. The WGSL route currently
admits only vertex, fragment and compute entry points, so it supplies no parsed
mesh metadata that could be checked here. Mesh pipelines validate supported Core
topology categories and native function stages; native MSL/metallib callers must
match the mesh output type to the descriptor. The backend does not pretend that
opaque library reflection proves this topology. Logical bind-group/index limits
also do not guarantee every resource combination: pipeline creation validates
combined per-stage native buffer, texture and sampler slot budgets.

Secondary recording, ray pipelines and traditional pipeline-statistics queries
retain their documented unsupported boundaries. They are not implemented with
CPU command replay, software tracing or fabricated query results. Synchronous
nextDrawable acquisition remains an explicit native behavior; this Windows-only
review provides no Apple runtime latency or GPU acceptance claim.

A generation change between Core surface preflight and final native handoff now
returns SurfaceOutdated. Reusing an already handed-off lease still returns
InvalidState. The final handoff checks the generation under the same lock used
by surface reconfiguration.

Metal TLAS rebuilds keep one accepted BLAS/contribution snapshot and one
owner-scoped recorded replacement. Cancellation leaves the accepted snapshot
unchanged; each native command-buffer completion handler holds its build inputs
until terminal execution and replaces the snapshot only on success. Checkpoints
also cover that metadata publication, independently of callback ordering. Old
BLAS owners and contribution metadata are not appended to the TLAS forever.

The final independent Metal source review found no further confirmed P0/P1/P2
issues after these changes. MSRV 1.87 Windows and aarch64-apple-darwin all-target
compilation checks pass. No Metal tests were executed; compilation and source
review do not establish native GPU, WSI, iOS linking or performance acceptance.

## WebGPU review corrections

Buffer support and creation share one native-usage admission function. Illegal
mapping combinations are rejected before browser object creation. The explicit
logical MAP_READ | QUERY_RESOLVE route uses non-mappable native resolve storage
with COPY_SRC plus a MAP_READ staging copy. Buffer objects retain their actual
native usage; mapping and readback use those facts rather than the logical usage.

Bind-group layouts and samplers retain backend-local descriptor metadata beside
their browser handles. Bind-group creation checks buffer usage, range, minimum
binding size, alignment and native binding-size limits; sampled/storage texture
usage, view shape and numeric/sample class; and sampler comparison/filtering
compatibility before issuing createBindGroup. Filtering layouts admit nearest
samplers, while non-filtering layouts reject linear samplers. Storage binding
sizes must be multiples of four; this rule is not imposed on uniform sizes.
These checks do not move platform handles or metadata into Core.

None and End completion requests retain a single batched queue submission.
Steps requests submit only at the requested boundaries and install a separate
onSubmittedWorkDone promise after each boundary. A trailing unrequested batch
still receives internal completion retention. Checkpoints report the exact
requested frontier; this observes the native single queue and adds no scheduler,
CPU command replay or artificial multi-queue model.

Fixed byte-patch/copy, indirect-count conversion and occlusion-segment reduction
compute kernels use device-local immutable pipeline caches. Per-operation masks,
values, chunk bases and counts are passed in retained uniform buffers. Partial
clear/resolve, color/depth blit, depth upload and stencil-copy render utilities
likewise cache only finite native format/sample/shape variants; colors, offsets,
rectangles and strides remain per-operation parameters. Mutable bind groups and
scratch resources remain local to recorded native work.

A Chrome diagnostic compares the same fixed byte-patch kernel with its device
cache retained versus explicitly invalidated before every operation. Across three
interleaved samples, median CPU recording times were 1.240 versus 1.950 ms for
16 patches, and 5.965 versus 7.815 ms for 128 patches. Every sample checks exact
GPU readback and untouched neighboring bytes. This isolates cache reuse; it is
not an old-revision or renderer frame-rate benchmark. Tightly packed indirect-count source
records use one native copy per chunk. Padded-stride packing still records
per-draw copies; a different GPU packing algorithm requires a measured candidate
before replacing that path.

The existing native texture usage lowering remains intact: where legal, logical
COPY_SRC resources receive the private native texture-binding capability needed
by GPU blit lowering, and renderable COPY_DST resources receive render-attachment
capability. These facts must be read alongside logical descriptors when auditing
utility paths. Cube view validation does not impose an invented six-layer base
alignment; the WebGPU standard constrains the cube layer count.

Secondary recording, multi-queue, pipeline statistics, persistent native pipeline
cache, mesh/ray, sparse/placed memory and external/native interop retain their
standard WebGPU capability boundaries. No CPU implementation fabricates support
for these operations.

Focused browser acceptance on Windows / Chrome 155 / AMD RDNA 3 (non-fallback
adapter) passes 42 tests, including native query readback, invalid descriptor
recovery, per-step checkpoints, byte-exact transfers, counted draws, split-pass
queries, partial render areas, MSAA, depth/stencil and canvas configuration.
ETC2 and ASTC subcases are skipped because that adapter does not expose those
format features; BC runs. Independent final source review confirms no further
P0/P1/P2 findings in the reviewed changes. This is browser/device-specific
acceptance, not a claim for every browser, optional feature or GPU.

## GL-family review corrections

Buffer support and allocation select the same ABI ceiling. Native buffer sizes
are limited by the actually loaded pointer-width allocation and mapping family,
with an i32 fallback when that family is absent. Persistent storage alone does
not lift the ceiling: mapping its full range must use the same admitted ABI.
WebGL2 uses its i32 bound for admission and validates conversion before creating
a browser buffer. Native allocation also converts before allocating a name;
subsequent failures delete unpublished objects. These are addressing limits,
not promises that physical memory can satisfy every admitted allocation.

Immediate Steps requests now capture completion at each requested successful
command/present close using the existing ImmediateBackend checkpoint operation.
Core retains only completion facts, including all returned lane points, and
returns them in request order with duplicates preserved. None/End and Deferred
recording are unchanged. Backends which return no close-time point retain their
previous conservative finish-time route. Core previously permitted later
conservative proof, so this fixes observation precision rather than claiming
that every old terminal-only result was incorrect. If observation fails after a
step has issued, the accepted prefix keeps that step and reports indeterminacy;
it does not falsely classify the step as having no native effect.

Desktop GL 4.3 or ARB_multi_draw_indirect can issue one native multi-draw call.
Both typed entry points and real two-record array/indexed probes must succeed
before that native capability is published. The ARB entry points use the
[unsuffixed names](https://registry.khronos.org/OpenGL/extensions/ARB/ARB_multi_draw_indirect.txt).
The reported count ceiling is GLsizei's signed-32-bit ABI limit, not a queried
hardware batching capacity. GLES 3.2 is not treated as core multi-draw-indirect;
its [EXT route](https://registry.khronos.org/OpenGL/extensions/EXT/EXT_multi_draw_indirect.txt)
is a distinct optional capability. Missing/unproved batch support retains direct
native single-indirect calls; arguments remain on the GPU. Indexed nonzero base
offsets retain their existing GPU rebase lowering without CPU decoding.

GLES counted draws use a context-owned fixed compute program with per-operation
uniforms. A transaction snapshots native program, generic/indexed SSBO, copy and
indirect bindings before mutation. It restores them on every return path and
deletes each successfully acquired scratch buffer, including partial allocation
failures. The program is deleted only from the final owner-thread teardown after
making that exact context current; a lost context discards its stale name without
calling into a different generation. No command stream is cached or replayed.

WebGL2 array/3D immutable storage retains its initialization workaround. On the
tested Chrome/ANGLE driver, removing the touches loses the first attachment
clear of untouched slices even though immutable-storage rules and GL error
checks appear satisfied. One 1x1 zero column covering all slices of a mip now
replaces individual per-slice uploads. This reduces initialization calls from
mips-times-slices to one per mip while retaining the demonstrated workaround;
it does not allocate a CPU-sized copy of the entire texture.

Focused acceptance covers native GL 4.6 / Radeon 780M array/indexed multi-draw,
tight/padded arguments, nonzero index-base GPU rebasing and GPU count outputs;
GL Steps coverage and readbacks on WGL, GLES and WebGL2; and GLES failure-injection
binding restoration/scratch deletion. The GLES runtime is LDPlayer Android 14,
whose Qualcomm/Adreno label identifies its translation layer, not the physical
GPU. Browser acceptance first passed 112 tests and exposed three initialization
regressions; the batched workaround then passed those three plus related
multiview/3D/array regressions. Ignored native fixtures are not counted as GPU
passes. This is a focused matrix, not all-driver conformance.

## DX12 indirect conversion and ray-cache boundary

WGSL vertex/index builtins require two root constants in addition to each
16/20-byte native argument record. GPU packing expands those records without
reading arguments on the CPU. Each private window respects the shader's 32-bit
local byte addresses and native dispatch limits while preserving u64 resource
offsets. GPU-count batches write a clamped count per window, including zero for
windows beyond the GPU's selected count. Raw SRV bases are aligned to 16 bytes;
the shader accounts for the permitted four-byte public offset. Native DXIL/HLSL
indirect routes retain their ordinary ExecuteIndirect ABI.

The current persistent cache stores ID3D12PipelineState objects through the
pipeline-library route, while DXR creates ID3D12StateObject. A live state-object
reuse cache or storing shader inputs does not by itself implement the existing
persistent native-cache contract. Passing a ray cache therefore remains explicitly
unsupported instead of being ignored. This does not mean DX12 can never support
ray-pipeline caching: Microsoft's newer
[State Object Database specification](https://github.com/microsoft/DirectX-Specs/blob/master/d3d/StateObjectDatabase.md)
includes raytracing descriptions and DXIL as part of Advanced Shader Delivery.
It is a distinct capture/precompilation mechanism, not the current PSO library;
the repository's pinned windows 0.62.2 bindings and backend do not expose/integrate
that route. It needs separate runtime/SDK admission and contract work rather than
a fabricated successful cache hit.

## Second DX12 runtime review corrections

Submission retirement follows the native queue that issued the work. Each used
lane signals its own terminal fence, retains its recorded owners against that
fence, and contributes its own End checkpoint. Requested Steps points likewise
name the issuing queue's fence. Retiring resources no longer inserts an implicit
Direct-queue wait for independent Compute work. Explicit plan waits, external
queue boundaries, and the caller's explicit wait_idle join retain their native
semantics. Query sealing follows the issuing lane; uncertain work remains
quarantined on that lane, while sparse mappings remain Direct-only. This uses
D3D12's [explicit multi-engine fence model](https://learn.microsoft.com/en-us/windows/win32/direct3d12/user-mode-heap-synchronization),
without adding resource scheduling or a hazard tracker.

A controlled real-GPU test holds Compute behind an imported native fence and
submits unrelated Direct work. The prior revision blocks the Direct checkpoint
until that gate opens; the corrected implementation completes its Direct
checkpoint and verifies the copied bytes before release. This demonstrates the
removed dependency, not a general frame-rate result. Example 23 selects the
checkpoint covering its Present frontier, which transitively follows its
explicit Compute-to-Graphics dependency, rather than relying on vector order.
Its DX12 and Vulkan presentations and particle readbacks were checked.

Shader validation and raster, compute, mesh and ray pipeline creation use a
bounded cold-object worker. Owned descriptors and native device leases cross the
thread boundary; command recording and submission stay on their existing paths.
The future admits work without blocking its executor. Full capacity yields
Pending with capacity notification, rather than rejecting a valid burst.
Cancellation drops unadmitted work, skips cancelled queued jobs and discards
in-progress results after native compilation returns. Native compilation itself
cannot be interrupted. Worker initialization failure and job panic become
structured errors. This is a host-side compilation facility, not a GPU scheduler.

Descriptor release preserves the ordered free-range vector and coalesces only
its immediate neighbors, avoiding the previous full-list drain and replacement
allocation per return. Allocation remains the same simple first-fit search.
Randomized release checks compare every state with the former algorithm, and a
real shader-visible heap churn test verifies complete recovery. These establish
mechanism and correctness, not a renderer throughput claim.

The WGSL indirect converter uses bounded GPU dispatches and restores the
caller's graphics state before ExecuteIndirect. Scratch buffers, argument/count
buffers and descriptor ranges follow the recording's native-fence lifetime.
GPU timestamp/readback guards cover tiny and large batches, indexed and padded
records, signed base-vertex bits, nonzero source/count offsets, and a count-selected
batch crossing the native dispatch boundary. The small native-copy fast path is
strictly bounded to at most three records; it does not scale with large batches.

Immutable packet/coverage preflight runs before the submission gate. Device-loss
observation, query admission, native handoff, query state commit and sealing stay
atomic inside the gate. Opt-in hardware diagnostics measure wait/hold intervals
for pre-recorded packets from one/four submitting CPU threads. Those intervals
include native handoff and do not imply a universal throughput gain. Production
builds carry no timing state or clock reads; no query reservation system or
additional submission scheduler is introduced.

Cold-worker validation includes full-capacity backpressure, waker replacement,
cancellation deregistration, panic resolution and lost-wake coverage. Each
pending future owns one removable admission registration, so replaced or cancelled
waiters do not accumulate until the next dequeue. A real-GPU burst above queue
capacity creates usable compute PSOs and verifies dispatch/readback after partial
cancellation. In an isolated diagnostic run, polling the first 32 PSO futures
occupied the caller for about 308 ms on the synchronous baseline versus 59 us
with the worker; this measures caller occupancy, not faster native compilation.

## Reusable Core plan validation and immediate prefixes

A compiled SubmissionPlan shares immutable metadata across its clones: step
identities, the presence of sparse steps and its topology-validation result.
Every execution still checks the actual DeviceId before consulting that result.
Device identity and its negotiated lane set are immutable under the existing
unsafe Device::from_native contract, so both successful topology validation and
its errors can be reused. Resource bindings, completion requests, submitted
payloads and recovery frontiers remain validated for each owning packet; this is
not a cache of resource state or recorded commands.

Immediate success and uncertainty results expose IssuedSteps, a read-only shared
prefix of the plan's cached step identities. Its slice view, borrowed iteration
and consuming copied iteration preserve the exact prefix without rebuilding a
Vec on each finish or requested checkpoint. Cloning a prefix shares its backing;
callers needing an independently mutable vector explicitly use to_vec(). Native
backend checkpoint operations continue borrowing a StepId slice. Failure effects,
accepted-prefix ownership and completion coverage are unchanged. Plan compilation
pays the identity-storage cost once; no Renderer frame API or scheduler is added.

DX12 retained work also has autonomous terminal-fence observation. A one-shot
Windows fence event and thread-pool callback releases completed native owners
when the application requested no completion points and performs no later RHI
operation. This breaks the Device/in-flight encoder strong-reference cycle.
Owners enter their lane stores before registration; already-completed fences
retire synchronously. The callback owns its temporary Device lease independently
of those stores and releases it after retirement. This is completion observation,
not command scheduling or periodic GPU polling.

Setup programs the fence event before arming the wait. Event/wait/programming
failures return Indeterminate after native handoff and place an error-only owner
in recoverable quarantine; explicit idle recovery releases it after completion.
Device removal preserves uncertain owners. Gated real-GPU tests drop all public
Device/resource/receipt handles before releasing native Direct/Compute gates,
then verify final ownership disappears without another RHI call. Separate
thread-local fault injection covers all three setup stages and explicit recovery.
Callback teardown follows the documented
[Windows wait-object lifetime](https://learn.microsoft.com/en-us/windows/win32/api/threadpoolapiset/nf-threadpoolapiset-closethreadpoolwait)
and avoids waiting for its own callback.

Final packing diagnostics used adjacent baseline/candidate pairs with identical
input records and GPU timestamp/readback checks. At 10,000 records, median warm
CPU recording was 1,298 versus 409 microseconds and the measured two-batch GPU
pass was 17.07 versus 8.46 milliseconds on Radeon 780M. Tiny-batch intervals
largely overlap; their fixed native-copy guard avoids unnecessary compute setup.
These are focused diagnostics, not application FPS or all-driver guarantees.
Private SRV windows also respect the guaranteed native view span. Arithmetic
beyond a 16-GiB source offset is checked separately; that is not a claim of a
physical multi-gigabyte allocation test.


## Vulkan submission and cold-object execution

Submission routing follows the packet's actual lanes, sparse operations and
explicit synchronization requirements. Device sparse/timeline capability alone
does not select multi-queue submission for graphics-only work. Each active lane
has its own terminal fence and retention set; End checkpoints cover their own
frontiers. No terminal graphics marker waits on an independent compute frontier.
A failed terminal marker preserves even owners already partitioned into lane
retirement sets; failure does not establish completion. Only explicit plan
edges and GPU tokens introduce cross-queue waits. `wait_idle`
observes a marker on every exposed native queue without adding a GPU queue join.

Timeline-capable devices own a persistent pair of raw queue semaphores and
monotonic counters. Packet wrappers and exported tokens retain the device but
the device does not retain those wrappers. Thus tokens remain valid across
packets and idle periods without recreating semaphores or introducing an
ownership cycle. Native queue access remains externally synchronized.

Host-coherent upload bytes are written before native submission, which supplies
the host-to-device visibility boundary. Buffer uploads record their copy and a
barrier restricted to the written destination range; they no longer surround
each write with global memory barriers. Dependencies on earlier GPU consumers
remain the caller's explicit barrier responsibility. Copy-offset alignment is
cached at device initialization rather than queried for each upload.

Device-local readback uses family-local, persistently mapped staging leases.
Both the ticket and recording retain the lease and its device, so cancelling
the ticket cannot recycle an in-flight page and an outstanding ticket remains
safe after the public device is dropped. The raw-page lease drops before its
device owner. Idle pages do not retain the device and the cache keeps at most
32 standard 4 MiB pages; oversized pages are released. Buffer capacity excludes
native memory-allocation padding. Host results copy bytes out only after
completion proof.

Shader translation and raster, compute, mesh and ray pipeline creation use a
bounded cold-object worker. Futures own cloned descriptors and retain their
device; a full admission queue applies asynchronous backpressure rather than
rejecting valid bursts. Cancellation removes its wake registration, and caller waker panics cannot
terminate the worker or overwrite a successful result. Native pipeline-cache
access remains synchronized. Resource creation and command
recording retain their existing native execution paths; the worker neither
records nor schedules GPU commands.


The focused Vulkan execution gate passed 41 library tests (including explicit
hardware fixtures) and 36 affected GPU integration tests with Khronos
synchronization validation. Controlled gates exercised graphics family 0 and
compute family 1: graphics completed while compute was held, and `wait_idle`
remained pending until compute was released. A terminal-marker fault injection
verified that already-submitted lane owners remain quarantined. Readback tests
covered ticket-only device lifetime, cancellation and completion-only reuse.
Example 23 presented 12 frames and its 1280 by 720 particle readback was opened
and checked. MSRV 1.87 all-target Vulkan compilation passed. These are focused
checks on the available device, not all-driver or frame-rate claims.
