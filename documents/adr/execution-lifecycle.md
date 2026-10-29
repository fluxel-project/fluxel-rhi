# Execution and lifecycle

This file groups related decision records to keep the index compact. Original ADR identifiers and status labels are retained; current normative contracts live in the linked design and testing documents.

## ADR-0003: Keep execution lowering serial until evidence justifies more

**Status:** Historical baseline. The current execution contract is in
[`design/core.md`](../design/core.md) and `crates/core/src/api/submission.rs`.

## Context

A RenderGraph execution plan describes portable dependencies, not a promise of
multiple queues, parallel recording jobs, command caching, or aliasing.

## Decision

The original native lowering used one logical queue, serial recording, and one
submission per frame. A shared queue-operation lock covers submit and
error-path idle waits. Same-state write dependencies remain semantic facts even
when a backend needs no state transition.

## Alternatives

- Treat graph parallelism as native queue parallelism.
- Add multi-queue, parallel recording, aliasing, or caches before a measured
  profile requires them.

## Consequences

Correctness is specified independently of performance strategy. Later lowering
optimizations must preserve plan ordering, visibility, completion, and lease
semantics and require their own evidence.

## Evidence

C03 in 0.1.2 exercised same-state overlapping writes; 0.1.3 added the shared
queue lock; 0.1.4 proved Raster→Compute→Copy in one submission. No release has
claimed multi-queue or parallel-recording support.

See the [Core design](../design/core.md#execution-and-completion) for the current execution contract.
RenderGraph architecture remains in the
[fluxel-rendering repository](https://github.com/fluxel-project/fluxel-rendering/tree/main/documents).

## ADR-0004: Quarantine accepted-unknown GPU work

**Status:** Superseded by the normative [RHI API v1 submission acceptance
contract](../design/core.md)
for 0.16.

**Supersession note.** This ADR preserves the evidence and safety concern from
the pre-v1 implementation history, but it does not define the 0.16 public API.
The accepted-unknown quarantine name and its generation-poisoning model are not
part of the v1 contract.

## Current RHI API v1 replacement

`Device::submit()` returns `Err` only when it guarantees that **zero native
work from the plan was accepted**. Once any native work has been accepted,
`Device::submit()` returns `Ok(SubmissionReceipt)`; a later submit failure,
device loss, presentation failure, or retirement outcome is reported through
the receipt's terminal `CompletionState` and, where applicable,
`PresentState`. Those terminal states exclusively own all post-acceptance
failure and retirement reporting. Accepted work remains retained until that
terminal lifecycle allows safe retirement.

## Context

A native submit error may occur after work has been accepted. Treating it as a
known rejection can release commands, staging memory, resources, or leases
while the GPU may still reference them.

## Historical decision

Distinguish known pre-submit rejection from accepted-unknown work and terminal
completion failure. Accepted-unknown work retains/quarantines all referenced
objects until safe retirement; it never publishes guessed outgoing state or a
ready snapshot.

## Alternatives

- Collapse all submit errors into one `Result` failure.
- Use queue-idle or a blocking drop as cleanup.

## Consequences

Pending operations are owning, non-blocking state machines. Renderer snapshot
reservations release before accepted work, but terminal failure or early drop
after acceptance poisons an affected generation when its state is unknown.

## Evidence

0.1.2 formalized structured completion and quarantine. 0.2.0 upload and
0.2.1–0.2.7 snapshot fault fixtures exercised partial acceptance, failure,
drop, and subsequent reuse behavior.

See the [Core design](../design/core.md#surface-and-resource-lifecycle) for the current lifecycle contract.
Renderer design remains in the
[fluxel-rendering repository](https://github.com/fluxel-project/fluxel-rendering/tree/main/documents).

## ADR-0009: Keep the resource floor closed and reuse stateful

**Status:** Historical (0.12–0.15). Superseded for all RHI semantics by
[Fluxel RHI architecture](../design/README.md).

## Context

The 0.12 resource closure needed resources to cross frames and, for compatible
compiled graphs, permits transient allocation reuse. Those two conveniences can
silently become unsafe if a caller treats a logical name as a physical object,
forgets a resource's real outgoing state, or frees an object after a submission
whose acceptance or completion is not known.

The same closure is needed on native and browser backends, but their useful
capabilities differ. A portable declaration must report an unsupported
operation precisely and before it creates browser or native work. None of this
requires a general shader, descriptor, or pipeline API.

## Historical decision

The 0.12 resource surface was a fixed, evidence-backed floor. It consisted of
the common fixed resource paths plus only the named closed compute/storage
recipes. Backend capability reports are observed facts; unsupported semantics
return structured `UnsupportedCapability` diagnostics and do not fall back to
emulation.

Persistent imports bind a provider-selected physical identity and generation,
actual incoming state, descriptor/usage facts, and a lease. Exports report the
actual outgoing state. A consumer must use that state rather than inventing a
convenient one.

Transient reuse is private executor policy. A slot is keyed by device and
compiled-graph generation as well as logical resource identity. It is eligible
only when whole-resource state can be carried exactly. On reuse, lowering uses
the remembered real outgoing state as the next frame's incoming state; partial
or mixed-state resources remain ineligible until the graph has a complete
subresource-state reuse model. Graph/device invalidation stops future checkout,
but does not destroy old slots until their work has terminal completion.

Pending and accepted-unknown work was quarantined. In particular, changing a
device or compiled-graph generation could not free a slot still owned by
pending or unknown work. Browser resource registries applied the same rule per
opaque key and device generation: tickets retained their resource leases until
safe retirement.

## v1 interpretation

This ADR no longer defines a resource floor, acceptance state, completion, or
loss protocol. Those are defined by the public RHI API and its
[architecture guide](../design/core.md).
The old fixed floor and accepted-unknown quarantine are historical evidence,
not APIs that implementations may continue to expose.

For future reuse and retirement, an executor retains an object through the
relevant `SubmissionReceipt` and its last-referencing `CompletionPoint`, until
that point reaches terminal completion or the owning device reaches terminal
loss. Loss terminates the complete `DeviceIdentity` / `DeviceGeneration`
domain; recovery obtains a new identity/generation domain and never mutates an
old public device through `generation++`. Compiled-graph generations and reuse
slot generations remain private executor bookkeeping, not RHI identity.

## Alternatives

- Reset every reused resource to `Undefined` and let the next plan transition
  from that fabricated state.
- Reuse allocations across arbitrary subresource histories without tracking
  their state.
- Destroy all old-generation resources immediately during graph/device loss or
  browser recovery.
- Advertise a broad portable resource profile and emulate unsupported WebGL2
  compute/storage operations.
- Generalize shader, pipeline, layout, or binding construction to express the
  new recipes.

## Consequences

Reuse can save compatible allocations without weakening RenderGraph state
semantics, but it deliberately excludes some otherwise-compatible resources.
Providers and browser registries carry generation/lease bookkeeping that a
stateless allocator would avoid. Capability matrices remain intentionally
asymmetric: WebGL2 rejects compute and storage with zero side effects, and a
storage-texture read is not promised on a backend unless that exact closed path
is supported.

Historically, this decision reinforced, rather than replaced,
[ADR-0004](execution-lifecycle.md#adr-0004-quarantine-accepted-unknown-gpu-work) for unknown completion and
[the historical ADR-0006](https://github.com/fluxel-project/fluxel-rendering/blob/main/documents/adr/0006-no-general-pipeline-yet.md)
for the closed-artifact boundary.

## Evidence

Historical portable contract tests covered capability diagnostics,
identity/state/lease validation, reuse eligibility, carry-over, invalidation,
and pending/unknown quarantine. Browser and native conformance fixtures tested
the implemented fixed paths; they did not turn the resource floor into a
general graphics API.

## ADR-0013: Make only future-event operations asynchronous

**Status:** Accepted

## Context

Thread safety and a future event are different properties. Turning ordinary
logical creation and recording into futures obscures ownership without adding
an asynchronous boundary.

## Decision

Adapter/device request, shader/pipeline compilation, surface configuration and
acquisition, buffer mapping, completion observation, and idle waiting use
futures. These operations can depend on an externally completed event.

Queue submission is synchronous: its result describes native acceptance,
including partial or uncertain acceptance, rather than GPU completion. Command
recording and immediate execution are also synchronous. Presentation executes
through the deferred or immediate execution contract; it is not universally an
async operation and does not establish GPU completion.

Readback copy encoding is synchronous. Waiting for its completion and mapping
the resulting buffer are separate future-event operations. Capability queries,
logical buffer/texture/view/sampler/binding creation, validation, statistics,
and diagnostics remain synchronous.

## Consequences

Backends may use immediate-ready implementations for an async operation, while
concurrent synchronous calls remain ordinary calls. A new async API requires a
real externally completed event.

## ADR-0014: Treat device loss as a terminal device identity state

**Status:** Accepted lifecycle principle; historical protocol target, not a
completed cross-backend Core contract.

The current authority is [the Core design](../design/core.md) and
[`core::api`](../../crates/core/src/api/mod.rs). Core uses `DeviceId` and does
not expose generic `status()` or `loss_info()` methods. The protocol below
records the original target, not APIs or uniform wake-up guarantees already
implemented by every backend. WebGL2's implemented terminal-generation behavior
is described separately in the current design. A shared loss-query API requires
agreement on native loss authority, observation semantics, and pending-operation
wake-up behavior across backends before it can be claimed as supported.

## Context

Native APIs cannot revive a removed device safely, and browser contexts have
the same ownership problem. A mutable generation counter would permit stale
objects to appear valid after recovery.

## Decision

Loss is terminal for one `DeviceIdentity`. It wakes pending completion,
readback, acquire, present, and idle futures; later operations return structured
`DeviceLost`; `status()` and `loss_info()` remain synchronously queryable. A
new request creates a distinct identity. There is no required `wait_lost()` API.

## Consequences

Callers recover by rebuilding device-scoped objects. Backends must route every
native loss observation through the same terminal authority and never publish
readback success after loss.

## ADR-0015: Freeze plan-scoped transient lifetimes with a dedicated fallback

**Status:** Historical design target. The current Core API does not expose the
`TransientLifetime` and `SubmissionPlanBuilder` described below; this record
does not claim an implemented backend capability.

## Context

Transient allocation affects resource lifetime and synchronization, so waiting
for a later allocator design would leave an API hole. Native heap/aliasing
mechanisms differ substantially.

## Decision

`TransientLifetime` uses an acquire `PlanPoint` and release frontier;
`SubmissionPlanBuilder` reserves batches and exposes a plan-scoped allocator.
Every backend implements `Dedicated`. `Aliasing` is advertised only after
physical placement, non-overlap proof, native barriers/dependencies, and
completion-safe retirement are complete.

## Consequences

WebGPU and GL remain correct without aliasing. DX12, Vulkan, and Metal can add
heap strategies later without changing public lifetime vocabulary.

## ADR-0021: Make executable shaders own immediate-data ABI

**Status:** Historical design target. `ShaderInterface`, `PipelineInterface`,
and `ShaderAbiVersion` below are not part of the current Core API.

## Context

`PipelineInterface` may intentionally be a compatible superset so renderers can
share it across pipelines. Native shader argument locations, however, are
compiled into an executable artifact. Deriving immediate-data argument presence
or indices from every range in a pipeline-interface superset lets an unused
range change a Metal `[[buffer(n)]]` ABI, or leave a shader-read argument
unbound.

## Decision

Each `ShaderInterface` declares the non-overlapping immediate byte intervals
its entry point reads. Pipeline construction requires every such interval to be
contained in a stage-visible `PipelineInterface` immediate range. The interface
remains a logical superset; it never assigns native immediate ABI.

For a fixed `ShaderAbiVersion`, a backend derives native immediate argument
presence, index and executable byte extent exclusively from the participating
shader interfaces and its frozen private ABI. Legal writes to a declared but
artifact-unused range have no executable observer and may be omitted by native
lowering.

## Consequences

Toolchains must reflect immediate reads into `ShaderInterface`, alongside
ordinary resource requirements. An artifact declaration is still a toolchain
contract; the RHI does not parse arbitrary native source to reconstruct it.
The shared pipeline validation prevents an incomplete immediate-layout contract
on every backend, while Metal additionally relies on it to keep direct argument
indices stable.
