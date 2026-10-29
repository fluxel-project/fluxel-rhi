# Fluxel RHI Architecture Decision Records

Related decisions are grouped by topic to keep the index concise. Original ADR numbers, statuses, and evidence notes remain in the records. Current normative behavior is defined by the crate-local README and DESIGN pages, Core API source, the [open TODO ledger](../../TODO.md), and [the verification guide](../verification.md).

## Backend and platform boundaries

- [ADR-0002: Native handle and unsafe containment](platform-boundaries.md#adr-0002-contain-native-handles-and-unsafe-in-the-rhi-implementation)
- [ADR-0011: GL family private layers](platform-boundaries.md#adr-0011-build-the-gl-family-through-three-private-layers)
- [ADR-0012: Portable execution boundary (superseded)](platform-boundaries.md#adr-0012-keep-rhi-limited-to-portable-execution)
- [ADR-0023: Host and RHI presentation integration](platform-boundaries.md#adr-0023-keep-host-windows-separate-from-rhi-presentation-integration)

## Execution and lifecycle

- [ADR-0003: Serial execution baseline (historical)](execution-lifecycle.md#adr-0003-keep-execution-lowering-serial-until-evidence-justifies-more)
- [ADR-0004: Accepted-unknown work (superseded)](execution-lifecycle.md#adr-0004-quarantine-accepted-unknown-gpu-work)
- [ADR-0009: Resource reuse safety (historical)](execution-lifecycle.md#adr-0009-keep-the-resource-floor-closed-and-reuse-stateful)
- [ADR-0013: Async operation boundary](execution-lifecycle.md#adr-0013-make-only-future-event-operations-asynchronous)
- [ADR-0014: Terminal device-loss principle](execution-lifecycle.md#adr-0014-treat-device-loss-as-a-terminal-device-identity-state)
- [ADR-0015: Transient allocation design target (historical)](execution-lifecycle.md#adr-0015-freeze-plan-scoped-transient-lifetimes-with-a-dedicated-fallback)
- [ADR-0021: Shader-owned immediate ABI design target (historical)](execution-lifecycle.md#adr-0021-make-executable-shaders-own-immediate-data-abi)

## Capability contracts

- [ADR-0016: Capability claims require complete lowering](capability-contracts.md#adr-0016-publish-capabilities-only-with-lowering-closure)
- [ADR-0019: Logical statistics principle (future)](capability-contracts.md#adr-0019-define-statistics-as-logical-observation-not-profiling)
- [ADR-0020: Optional feature families](capability-contracts.md#adr-0020-admit-optional-feature-families-as-complete-portable-contracts)

## Presentation and evidence

- [ADR-0005: Earlier GPU conformance evidence workflow (superseded)](presentation-and-evidence.md#adr-0005-require-gpu-conformance-evidence-for-native-correctness-claims)
- [ADR-0008: Earlier native platform test gates (historical)](presentation-and-evidence.md#adr-0008-execute-platform-specific-test-paths-natively)
- [ADR-0017: Presentation frame lifecycle](presentation-and-evidence.md#adr-0017-keep-presentation-frames-separate-from-textures-and-completion)
- [ADR-0022: Hardware conformance evidence](presentation-and-evidence.md#adr-0022-separate-portable-tests-from-hardware-conformance-evidence)
