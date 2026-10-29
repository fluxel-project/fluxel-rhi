# Core Design

`fluxel-rhi-core` is the contract layer between a renderer that has already planned GPU work and a concrete GPU backend. It defines the portable object model, validates facts that can be checked without a driver, and presents one typed API for execution, surfaces, synchronization, and completion.

It is not a portable GPU implementation. DX12, Vulkan, GL, WebGPU, and Metal keep their native objects, API calls, threading rules, and feature limits in their own crates. Core makes those differences visible as capabilities and structured errors rather than flattening them into a false common denominator.

Core is designed first for RenderGraph and rendering-pipeline implementations.
It keeps the shared execution contract thin and low-cost, preserving native
capabilities and exposing execution facts needed for graph planning.
Application-level developer convenience is secondary to predictable execution
of already planned work.

## Position in the RHI

```text
RenderGraph / Renderer
  owns scheduling, hazards, transitions, frame policy, and retirement
                              |
                              v
                        fluxel-rhi-core
  typed API, validation, object provenance, execution protocol, completion
                              |
                              v
                          backend SPI
  associated native types and lowering traits implemented by each backend
                              |
                              v
             DX12 / Vulkan / GL family / WebGPU / Metal
```

The facade crate only selects and reexports concrete backends. Core does not perform runtime backend dispatch: `Device<B>`, `Buffer<B>`, `Texture<B>`, and encoders retain the backend type `B`, so hot paths use static dispatch and do not require a backend enum, `Any`, or downcasts.

## What Core owns

The public namespace, [`src/api/`](src/api/), contains the API a renderer uses:

- **Foundation and handles** define instances, adapters, devices, lanes, capabilities, errors, and typed wrappers for GPU objects.
- **Creation, resources, shaders, pipelines, and bindings** describe buffers, textures and views, layouts, shader artifacts, and pipeline construction.
- **Encoders and passes** express copies, transitions, raster and compute passes, query use, optional mesh/ray work, and capability-specific commands.
- **Submission and execution** represent the caller's ordered plan, native recording/issue protocol, partial-effect outcomes, and completion requests.
- **Surfaces, mapping, queries, and interop** expose presentation leases, asynchronous observations, and narrowly scoped backend-native access.

Core owns the common meaning of those operations and validates their portable preconditions. It does not own a window or canvas, render-graph construction, resource-use inference, hazard analysis, automatic transitions, frame pacing, or resource-retirement policy. Those are intentionally renderer or host responsibilities.

## Architecture inside the crate

```text
api/ facade methods
  validate typed descriptions, ownership, range and state preconditions
             |
             +--> Backend associated native object types
             |
             +--> backend/ lowering traits
                    resources | commands/passes | submission | completion
             |
             v
         backend-private native implementation
```

`api/` is public. `backend/` is `#[doc(hidden)]`, but is the implementation boundary consumed by backend crates. The split prevents native types such as COM interfaces, Vulkan handles, GL contexts, browser objects, and Objective-C objects from entering renderer-facing resource descriptions.

`Backend` in [`src/backend/foundation.rs`](src/backend/foundation.rs) assigns associated types for every native resource, encoder, surface image, completion point, and uncertainty record. A backend also chooses static adapters for optional domains such as compute, mesh, ray, raster extras, and memory observation. An unsupported domain uses the shared `Unavailable` adapter; backends do not implement a fake native route just to fail later.

`CommandBackend`, `RasterBackend`, and related traits in [`src/backend/operations.rs`](src/backend/operations.rs) lower validated portable commands. `DeferredBackend` and `CompletionBackend` in [`src/backend/submission.rs`](src/backend/submission.rs) own native submission and progress observation. Core supplies the public façade around those traits; the backend is responsible for native lifetime, synchronization, and unsafe invariants at the lowering boundary.

## Object provenance and lifecycle

Every opened device receives a `DeviceId`. Public resource wrappers retain that identity together with their backend type; plan lanes, submission indices, surface objects, and steps carry related identity. A `Buffer<B>` can never be passed to a `Device<C>` at compile time, and a `Buffer<B>` from a different `Device<B>` is rejected with `Error::WrongDevice` before a native call. Backends add their own slot/generation protection when native handle reuse makes it necessary.

The normal creation path is:

1. An `Instance<B>` enumerates adapters and opens one `Device<B>` with a required-feature request.
2. The backend returns its native device plus `DeviceCapabilities` measured for that exact device and enabled feature set.
3. `Device::from_native` derives aggregate facts and asserts that every advertised optional capability has a matching static lowering adapter.
4. API creation methods validate portable descriptions, invoke the resource SPI, and wrap the resulting native object with the device identity.

Capability values are facts about an opened device, not promises based only on a Cargo feature or backend name. They include native features, shader/artifact routes, binding shapes, and limits. `Error::Unsupported` means the selected route is absent; it is distinct from wrong-device, invalid-state/range, device-lost, and surface-lifecycle errors.

Core cannot prove that a renderer's transition sequence is correct. It checks the declared operation shape and provenance, while the renderer keeps the graph-level resource state and hazard knowledge. This distinction allows a backend to preserve native state models without Core inventing a second one.

## Execution models and operation flow

The caller first compiles its resource and dependency decisions into a `SubmissionPlan<B>`. Each plan names lanes, ordered `StepId`s, step kinds, and predecessor waits. At execution start, Core checks plan ownership, lane membership, step order, and completion-request shape. It does not reorder steps, infer dependencies, add barriers, or turn waits into memory visibility.

```text
renderer builds SubmissionPlan + explicit transitions
                 |
                 v
Device::begin_execution(plan, completion)
                 |
                 +-- deferred backend: record native encoders, finish, submit
                 |
                 +-- immediate backend: issue each command on its context
                 |
                 v
Execution::finish() returns accepted prefix and completion evidence
                 |
                 v
renderer observes completion, then decides retirement
```

For deferred backends, `Execution` stores backend-native recordings and later hands one packet to `DeferredBackend::submit`. It never builds a portable CPU command stream for capture or replay. For immediate backends, notably GL, `ImmediateSession` validates a plan once and issues each operation in order on the current context thread. The two paths deliberately differ in when a native effect becomes irreversible, while preserving one public description of steps, errors, and completion.

`Execution::finish` is required even after a command error because an earlier prefix may already have native effects. Outcomes distinguish a clean rejection, an accepted/issued prefix, and an indeterminate native effect. Backends retain the native objects required by issued work until completion or a terminal outcome establishes their disposition. A completion observer reports progress; it neither waits implicitly nor authorizes renderer resource retirement.

## Surface leases

Core models an acquired presentable image as an exclusive native lease. Surface configuration publishes a new generation only after native configuration succeeds. A generation-bound image can be presented or abandoned exactly once; an image from an earlier configuration is rejected rather than being presented to a resized or lost surface.

`NativeImageLease` is the narrow backend contract for this rule. It combines generation validation, irreversible native handoff, and abandonment without making the host transfer ownership of its window or canvas. The host remains responsible for resize, visibility, foreground/background, and restoration; the platform adapter translates those events into surface operations.

## Synchronization and observations

Explicit transitions and cross-lane waits are input to the plan. A wait orders execution only; it does not manufacture memory visibility or infer a missing barrier. Completion checkpoints, mapping, query publication, and CPU reads are separate operations with separate native evidence. This avoids treating a successful submit, process exit, or presentation call as proof that GPU work completed.

`Device::stats()` is deliberately narrow: it counts successful Core requests and handoffs using atomic snapshots. It is neither a driver query nor a memory, residency, presentation, or profiling report.

## Design constraints for new API

Add a shared operation only when it has one meaningful contract across actual consumers. Define its ownership, device identity, capability evidence, error path, and completion effect before adding backend lowering. Prefer Rust types, borrows, and exclusive leases to runtime checks. If an operation has no shared meaning, keep it optional or backend-specific instead of weakening the Core contract.
