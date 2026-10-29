# Core Design

`crates/core` defines the portable, statically typed GPU execution contract and
the implementation boundary that backend crates must satisfy. It gives
renderers one meaning for resources, operations, capabilities, synchronization,
errors, surfaces, and completion without claiming that every backend supports
every operation.

## Responsibilities

Core owns:

- Caller-facing API types, resource handles, device methods, and structured
  errors in `src/api/`.
- Backend implementation contracts in `src/backend/`.
- Shared identity checks and contract utilities used by backends.
- Common meanings for deferred execution, immediate execution, presentation,
  synchronization, and completion observations.

Core does not own backend-native resource representation, host window or
browser lifecycle, render-graph scheduling, resource-state planning, hazard
analysis, frame pacing, or renderer resource retirement.

## API and backend SPI

`api/` is the renderer-facing contract. Its modules group resource creation,
shader and pipeline descriptions, passes and encoders, submission plans,
execution outcomes, synchronization, mapping and queries, surfaces, and
optional operation domains. The backend SPI mirrors the same operations and
observable results so callers do not see backend-specific variations in shared
concepts.

Native objects remain associated backend types. COM interfaces, Vulkan
handles, GL contexts, browser objects, and Metal objects do not become portable
Core resources. Unsafe SPI calls mark the boundary where each backend must
uphold native lifetime and synchronization rules; Core checks common
preconditions but cannot prove GPU correctness by itself.

## Identity, capability, and error model

Resources carry their backend type and device identity. The type prevents
cross-backend mixing; runtime identity and generation checks reject stale or
cross-device objects within one backend before native calls. Structural checks
cover declared ranges, operation shape, device ownership, and state
preconditions.

Optional operation domains are represented explicitly. A missing capability
is not implemented as a successful no-op. Capability queries describe the
route available on the selected device, including enabled native features,
shader/artifact path, binding shape, and relevant limits. Structured errors
separate unsupported work from invalid identity or state, range failure,
device loss, and surface lifecycle failure.

## Execution and completion

The caller constructs an unsafe `SubmissionPlan` after resolving its own graph
dependencies. The plan records ordered steps, lanes, work kinds, and waits on
prior steps. Core validates device ownership, lane declarations, and
predecessor indices when execution begins. It does not sort steps, infer
resource hazards, synthesize transitions, or add a second scheduler.

Deferred backends retain native recordings and bindings for submission.
Immediate backends issue native operations during encoding. Neither path stores
a portable CPU command stream for replay. Their distinct SPI traits preserve
the different points at which work becomes irreversible.

Outcomes distinguish accepted steps, affected native work, and GPU completion.
Failures can leave no effect, partial work, or an unknown native effect.
Completion points are observations of native progress, not queue waits and not
resource-retirement policy. Backends keep native objects needed by issued work
alive until completion or a terminal failure establishes disposition. Dropping
a completion observer does not cancel submitted GPU work.

## Surface and resource lifecycle

The host owns the window, canvas, and their lifecycle. A Core `Surface` owns
the device presentation path associated with that host resource. Configuration
publishes a new generation only after native configuration succeeds. An
acquired surface image is an exclusive, generation-bound lease; presentation
or abandonment consumes it. Stale use after reconfiguration or loss is
rejected.

Core does not define frame pacing, drawable acquisition policy, resize policy,
or automatic recovery. A platform adapter translates host lifecycle events to
surface operations; the backend implements the native lease and reports the
result.

## Observability

`Device::stats()` counts successful Core requests and handoffs. These are
requested-operation counters, not driver-native state changes, residency,
memory usage, frame boundaries, display completion, or a GPU profiler. Reading
a snapshot performs atomic loads and does not issue native GPU queries.

## Contract review rule

Design the smallest public operation and ownership model that expresses a real
consumer need. Prefer Rust ownership, borrowing, lifetimes, and typed states to
runtime checks where possible. Review construction steps and error paths from
the caller's perspective. When an operation cannot be represented uniformly,
keep it optional or backend-specific rather than weakening a shared meaning.
