# Fluxel RHI Architecture Decision Records

These records preserve decisions owned by `fluxel-rhi`. They constrain the
portable GPU execution contract, backend containment, capability claims, and
RHI conformance evidence. Renderer and RenderGraph decisions remain in the
[fluxel-rendering repository](https://github.com/fluxel-project/fluxel-rendering/tree/main/documents).

## Index

- [ADR-0002: Contain native handles and unsafe in the RHI implementation](0002-rhi-unsafe-containment.md)
- [ADR-0003: Keep execution lowering serial until evidence justifies more](0003-serial-execution-lowering.md)
- [ADR-0004: Quarantine accepted-unknown GPU work](0004-accepted-unknown-quarantine.md) — historical
- [ADR-0005: Require GPU conformance evidence for native correctness claims](0005-gpu-conformance-evidence.md)
- [ADR-0008: Execute platform-specific test paths natively](0008-native-platform-test-gates.md)
- [ADR-0009: Keep the resource floor closed and reuse stateful](0009-resource-floor-and-reuse-safety.md) — historical
- [ADR-0011: Build the GL family through three private layers](0011-gl-family-three-layer-boundary.md)
- [ADR-0012: Keep RHI limited to portable execution](0012-pure-rhi-execution-boundary.md)
- [ADR-0013: Make only future-event operations asynchronous](0013-async-operation-boundary.md)
- [ADR-0014: Treat device loss as a terminal device identity state](0014-terminal-device-loss.md)
- [ADR-0015: Freeze plan-scoped transient lifetimes with dedicated fallback](0015-plan-scoped-transient-allocation.md)
- [ADR-0016: Publish capabilities only with lowering closure](0016-capability-claims-require-lowering-closure.md)
- [ADR-0017: Keep presentation frames separate from textures and completion](0017-presentation-is-not-a-texture-view.md)
- [ADR-0018: Preserve RHI observability without owning capture runtime](0018-capture-observability-without-capture-ownership.md)
- [ADR-0019: Define statistics as logical observation, not profiling](0019-portable-logical-statistics.md)
- [ADR-0020: Admit optional feature families as complete portable contracts](0020-optional-feature-family-admission.md)
- [ADR-0021: Make executable shaders own immediate-data ABI](0021-shader-owned-immediate-abi.md)
- [ADR-0022: Separate portable tests from hardware conformance evidence](0022-hardware-conformance-evidence.md)
- [ADR-0023: Keep host windows separate from RHI presentation integration](0023-host-rhi-integration-boundary.md)
