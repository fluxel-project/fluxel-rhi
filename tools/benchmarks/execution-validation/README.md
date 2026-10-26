# Execution validation diagnostic

Run from the repository root:

```powershell
cargo build --release --manifest-path tools/benchmarks/execution-validation/Cargo.toml
cargo run --release --manifest-path tools/benchmarks/execution-validation/Cargo.toml
```

The executable emits CSV samples for synthetic single-lane command plans with no waits. The `validation` rows measure Core plan validation. The `immediate_steps` rows run an actual public `ImmediateSession`: one Commands encoder is opened and closed for every plan step, and the mock backend returns one completion checkpoint for every requested step.

The immediate workload reports 1k and 10k plans for two request densities: every step and every 16th step. It exercises Core's plan-indexed Steps membership, sparse checkpoint storage, and final receipt construction. The mock has no GPU or driver work, so the numbers only establish CPU scaling of this Core path. They do not measure a real RenderGraph, native GL calls, backend submission, or GPU work, and do not on their own establish a product bottleneck. Its only dependency is the local Core crate; no additional dependencies are introduced.
