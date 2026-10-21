# Execution validation CPU diagnostic

Decision: retain the current implementation. No representative RenderGraph workload was supplied, and this synthetic measurement does not establish a product bottleneck or compare implementations. Reconsider only with a real workload and profiling evidence. This is a diagnostic, not a release gate or GPU throughput measurement.

Measured `SubmissionPlan::validate_for_execution` on an inert mock device, using one graphics lane and ordered command steps. The no-wait guard has no waits; the chain has each step after the first wait for its immediate predecessor. Each plan is constructed outside the timed loop. Both the plan input and result pass through `black_box`; each result is unwrapped. Four sizes run sequentially, with 1,000 warmup validations per case, then 21 paired samples with alternating case order. Each sample uses max(2,000, 2,000,000 / steps) validations. Timings are wall-clock elapsed divided by iterations; sample min/max show observed spread, not confidence intervals. A separate release build succeeded before this final measurement; its log is `validation-build.log`. No allocation or profiler data was collected. There is no native call and no mock SPI call count in this benchmark; existing execution contract tests cover mock calls separately.

| Steps | Case | Median ns | Min ns | Max ns |
| ---: | --- | ---: | ---: | ---: |
| 2 | No wait | 3.246 | 3.227 | 3.353 |
| 2 | Chain | 3.706 | 3.652 | 3.924 |
| 32 | No wait | 31.338 | 31.128 | 57.445 |
| 32 | Chain | 44.102 | 42.326 | 55.075 |
| 256 | No wait | 246.953 | 245.238 | 447.197 |
| 256 | Chain | 350.640 | 337.417 | 442.985 |
| 1024 | No wait | 975.150 | 966.400 | 1085.500 |
| 1024 | Chain | 1512.900 | 1484.100 | 1549.350 |

Environment: Windows 10.0.26300.0, AMD Ryzen 7 8845H, 8 cores / 16 logical processors, Windows High Performance power scheme. `rustc 1.97.0-nightly (507271bc1 2026-05-17)`, LLVM 22.1.4, `cargo 1.97.0-nightly (4d1f98451 2026-05-15)`, host `x86_64-pc-windows-msvc`, release profile. Machine load and CPU frequency were not controlled. This run is rapid diagnostic evidence only; timings are specific to this synthetic plan and environment.

Git HEAD `c9d25e52b6d620a63f58216cc5b376791a34bc7d`, dirty working tree. Measured `crates/core/src/api/submission.rs` SHA-256 `7cfcace0a43a3704f6ec11385d7e0a538fe86fb70188f1a491357a576a324311`. Harness `tools/benchmarks/execution-validation/src/main.rs` SHA-256 `0340943da70aab1a39f0c92b2e1b083070e73ae975d65def939ca1ca789e0a78`. Raw `validation-samples.csv` SHA-256 `70a9ce9240f0c9d90c5c016bc4ff798a6a7ffe251f17f7b8dfff2adc4772b240`. Build log SHA-256 `f7b0a5d8cd04adbea3f2fe45a93b184ecf2f50870643372d515aa1ee49486ce6`.

Reproduce from repository root: `cargo run --release --manifest-path tools/benchmarks/execution-validation/Cargo.toml`. Full correctness, failure paths, other wait patterns, multiple lanes, ImmediateSession and Execution wrapper costs, real GL calls and GPU behavior are outside this run. The benchmark checks only that each synthetic valid plan is accepted. Single-lane plans do not represent general RenderGraph workloads.
