# Execution validation diagnostic

Run from the repository root:

```powershell
cargo build --release --manifest-path tools/benchmarks/execution-validation/Cargo.toml
cargo run --release --manifest-path tools/benchmarks/execution-validation/Cargo.toml
```

The executable emits CSV samples for synthetic single-lane command plans with no waits or a chain of predecessor waits. It measures Core plan validation on the CPU. It does not measure a real RenderGraph, the execution wrapper, native GL calls, or GPU work. Results do not establish a product bottleneck. Its only dependency is the local Core crate; no additional dependencies are introduced.
