# Execution failure boundary and lifetime evidence

Candidate: worktree based on `c9d25e52b6d620a63f58216cc5b376791a34bc7d`,
identified by [final Rust fingerprints](sources/final-rust.json). No native
command lowering, shader, scheduling, or normal completion policy changed.

## Contract changes

An identifiable unsuccessful immediate slot reports StepEffect None, Partial,
or Unknown. This metadata is recorded before emergency completion evidence is
created: failure to prove earlier work complete does not turn a known clean
current rejection into an unknown current effect. Completion creation failure
after all successful steps does not invent a failed slot. Legacy outcome views
and deferred results remain conservative when their data cannot identify a slot.
Raw submission enums, recovery ownership and native SPI signatures are retained;
new facts/outcome struct literal constructors must supply the failed-step field.

A clean Present rejection remains retryable and returns the binding. Ending
without retry preserves its original error; successful retry clears it. The
shared example helper retains affected image leases, returned resource values,
completion and uncertainty tokens through its failure-only drain. Scene owners
remain the caller's lifetime responsibility. No queue of CPU instructions,
replay, scheduler, normal-path idle, or per-step fence was added.

## Verification

| Check | Result | Scope |
| --- | --- | --- |
| Core unit tests | 38 passed | Exact failed slot effects, prior-prefix checkpoint uncertainty, completion failure after all steps, rejection/retry and existing contracts |
| New helper lifetime tests | 4 passed | Native issue before error, image/value/checkpoint/uncertainty Drop after idle, exactly one uncertain handoff, no successful-path idle |
| Existing host/SPI tests | 7 passed | Submission ownership, partial/uncertain drain, cancellation/fail-stop and native signal bridge |
| GPU fixtures explicitly selected with --ignored | 5 passed | Legacy DX12/Vulkan plan tests and unified DX12/Vulkan/WGL tests, exact readback and WGL clear/Present smoke |
| Builds | Workspace and browser all-target checks passed | Checks preceded only final documentation/formatting edits; no platform implementation changed |
| Review | Independent read-only review found no blocker | Not a substitute for fault injection or GPU evidence |

GPU tests marked ignored require explicit hardware selection. Both this packet
and the [migration packet](../execution-api-2026-10-21/README.md) contain actual
execution logs; the attribute is not a claim that the tests were never run.
Native visual captures blocked by the earlier Windows lock screen remain
unverified; this change adds no new visual acceptance, Android matrix, Metal
hardware, cross-queue, device-loss or real RenderGraph integration claim.

## CPU diagnostic decision

Retain the simple validation implementation. The standalone
[diagnostic harness](../../../tools/benchmarks/execution-validation/README.md)
measures single-lane synthetic plans only, with no waits and with a predecessor
wait chain. Raw paired samples, observed spread, warmup, toolchain, machine and
power configuration are preserved in [the diagnostic record](diagnostics/validation-summary.md).
The 1024-step medians on this run were about 0.98 microseconds without waits
and 1.51 microseconds for a chain. These are not confidence intervals, native
GL timings, Execution wrapper measurements, or evidence of a product bottleneck.
Machine load/frequency were uncontrolled. No cache, registry or optimization
was introduced based on these diagnostic numbers.

## Reproduction

```powershell
cargo test -p fluxel-rhi-core --lib
cargo test -p fluxel-rhi --no-default-features --test frame_submission --test execution_failure_lifetimes
cargo test -p fluxel-rhi --features vulkan,native-gl-wgl --test unified_execution_gpu --test execution_plans_gpu -- --ignored --nocapture
cargo run --release --manifest-path tools/benchmarks/execution-validation/Cargo.toml
```

`files.json` indexes preserved bytes. Full build output and the diagnostic smoke
build log remain in the local working archive. This is change-specific evidence,
not a release or universal correctness/performance certification.
