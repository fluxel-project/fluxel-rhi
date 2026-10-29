# Presentation and conformance evidence

This file groups related decision records to keep the index compact. Original ADR identifiers and status labels are retained; current normative contracts live in the linked design and verification documents.

## ADR-0005: Require GPU conformance evidence for native correctness claims

**Status:** Superseded by [ADR-0022](presentation-and-evidence.md#adr-0022-separate-portable-tests-from-hardware-conformance-evidence)
and the current [verification guide](../verification.md). This record describes an older
`ExecutionPlan` and release-artifact workflow; it does not require a GitHub
Release or attachments for the current repository.

## Context

Compilation, mocks, and CPU protocol tests cannot prove that DX12 and Vulkan
commands, barriers, shaders, and readback agree with portable semantics.

## Decision

Claim native correctness only when the same compiled ExecutionPlan runs on
DX12 and Vulkan with Required validation and an independent CPU oracle.
Artifacts identify the exact commit, backend, hardware/driver, inputs,
expected/actual result, completion, outgoing state, and diagnostics.

Release conformance is run from a clean Windows `x86_64-pc-windows-msvc`
checkout. The runner rejects a conflicting `CARGO_BUILD_TARGET`, records the
checked-out 40-character SHA, and writes the command, toolchain, target,
environment, GPU identity, timestamps, exit status, and observed case results
alongside the run log. These local artifacts are release evidence, not
source-controlled claims of success. After the matching annotated tag is
pushed, the artifacts are attached to that tag's GitHub Release; their durable
asset URLs close the evidence chain without changing the tested commit.

## Alternatives

- Treat CI compilation, TestRhi, or one backend demo as conformance.
- Let readback assume a convenient state instead of consuming an export's
  reported outgoing state.

## Consequences

Hardware fixtures are distinct from ordinary tests and are rerun on the final
commit before release through the release conformance runner. CPU and compile evidence remain
useful but are labeled as such; they do not substitute for native oracle
evidence. A release is not tagged when either backend's required fixture is
unavailable or fails.

## Evidence

0.1.2 C01–C03, 0.1.3 K01–K02, and 0.1.4 R01/R02/X01 established the pattern.
0.2.0–0.2.7 extended it through U01–U08 and exact-SHA release reruns.

See the [verification guide](../verification.md) for the current test-evidence boundary.

## ADR-0008: Execute platform-specific test paths natively

**Status:** Historical platform baseline. Current gates are described in the
[verification guide](../verification.md); the stub and feature combinations below refer
to the older repository layout.

## Context

Conditional compilation can leave platform fallbacks and feature combinations
untested. The Linux all-features Clippy failure in the non-Windows RHI stub
showed that Windows cross-compilation does not exercise the Linux path.

## Decision

Use Windows MSVC as the Windows development and native GPU target. Run Linux
workspace tests and `clippy -D warnings` on a native Linux environment such as
WSL2/Ubuntu, not merely as a Windows cross-target. Test reachable default,
all-feature, no-default-feature, and test-support combinations, including
structured platform-stub errors. Android support, when introduced, requires an
explicit target/NDK check and executable emulator or device tests for platform
code.

## Alternatives

- Treat a Windows build as coverage for `cfg(not(windows))` code.
- Require only that platform stubs compile.

## Consequences

CI gains a Windows workspace gate in addition to backend feature gates, while
Linux remains a native gate. Platform tests distinguish `BackendDisabled` from
`PlatformUnsupported`; GPU correctness still requires target hardware.

## Evidence

The 0.2.8 implementation removed the four unused non-Windows test-support stub
warnings under Linux all-features Clippy and added Windows workspace and
no-backend gates. The no-backend gate verifies the structured
`BackendDisabled` result for both native backend requests.

See the [RHI design overview](../design/README.md#system-boundary) for the current platform policy.

## ADR-0017: Keep presentation frames separate from textures and completion

**Status:** Accepted

## Context

Swapchain images, WebGPU current textures, CAMetalDrawables, and GL default
framebuffers do not share normal texture ownership. Present timing also differs
from GPU completion.

## Decision

`SurfaceImage` is distinct from `Texture` and `TextureView`; an acquired image
is non-cloneable and has one lifecycle. `PresentBinding` transfers its exclusive
lease into a submission plan. A rejected handoff can return the lease; an
unhanded lease is abandoned through its native drop path. Native present
acceptance, GPU completion, and present outcome remain separate observations.

## Consequences

Backend lowering can normalize attachments privately without leaking a
swapchain model. Backends own the safety of their native lease abandonment and
configuration-generation checks.

## ADR-0022: Separate portable tests from hardware conformance evidence

**Status:** Accepted

## Context

Portable positive, negative, and boundary tests establish the RHI contract,
but cannot prove that a backend's native lowering, synchronization, format
route, or device-loss handling works on a real adapter. Conversely, a window
that appears correct is weak evidence: it is difficult to automate and often
misses readback, ordering, and lifetime errors.

## Decision

Keep evidence in three deliberately separate tiers:

1. **Portable contract tests** are deterministic CPU-facing unit tests. They
   validate public API rules and must not require a native adapter.
2. **Headless hardware conformance tests** record one focused RHI workload on
   a selected backend, read its observable output back, and assert it on the
   CPU. They are the required native evidence for a capability claim. Cases
   should isolate a route where practical: buffer copy, compute, texture
   upload/readback, raster output, depth/stencil, resolve, indirect, query,
   compressed format, and loss/completion behaviour.
3. **Presentation smoke tests and examples** exercise acquire, present,
   resize, retirement, and shutdown with a real surface. They may be visually
   inspected, but do not replace tier 2. Larger examples adapt representative
   workloads rather than importing another framework's object model.

Shared tier-2 cases run against every backend that publishes the relevant
fact. Backend-private fixtures may create native windows, contexts, or
adapters, but those details never become public RHI types.

Every run records the crate revision, target, backend, adapter/driver and
enabled feature set. Results are classified precisely:

- **Hardware pass:** the asserted workload ran on the named real adapter.
- **Unsupported:** the capability fact correctly rejected the case before
  native work; this is not a pass for that capability.
- **Skipped:** required platform, adapter, browser, surface, or permission was
  unavailable; this supplies no conformance evidence.
- **Failure:** setup, lowering, validation, completion, readback, or assertion
  failed. A failure is never reclassified as unsupported after work begins.

Mock, static, compile-only, and pure encoder tests remain valuable regression
tests, but must be labelled as such and never reported as hardware evidence.

## Consequences

Capability closure requires at least one focused headless hardware assertion
on each platform/backend whose fact is published. Surface support additionally
requires an appropriate presentation smoke result. Windows, Android, browser,
and macOS evidence are separate: success on one cannot stand in for another.

The suite grows from small mechanical cases first; examples are a second-layer
combination workload, not a replacement for conformance cases or a promise of
wgpu API compatibility.
