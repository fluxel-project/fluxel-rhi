# RHI testing

The workspace contains Core, DX12, Vulkan, WebGPU, the GL family, an in-progress Metal backend, and the typed facade. Metal source compilation is not a substitute for macOS SDK linking or Apple GPU execution.

## Test source layout

Keep unit-test source under each crate's `src/tests/` tree and integration-test
targets under the crate-root `tests/` directory. Unit tests that need private
module access are declared from the module they exercise with `#[cfg(test)]`
and an explicit `#[path]` into `src/tests/`; this preserves their Rust module
scope while keeping test source out of implementation directories. Preserve
the source-module hierarchy under `src/tests/` where practical. Keep shared
integration-test helpers in nested `tests/<group>/` modules so Cargo does not
discover them as separate test targets.

Run portable checks and build the numbered examples from the workspace root:

```powershell
cargo test -p fluxel-rhi-core
cargo test -p fluxel-rhi-metal
cargo check --workspace
cargo build -p fluxel-rhi --features examples,vulkan --examples
```

Exercise Vulkan ray-pipeline creation with validation layers disabled as well
as enabled. Cover stages without specialization overrides and stages with
overrides: layer-enabled success alone can hide a native creation failure.
Verify the example's GPU readback and visible output after pipeline creation.

## Compiled execution plans

Core's execution fixtures consume hand-authored compiled plans; they do not
build a RenderGraph. They check unchanged step/wait/payload delivery, plan reuse
with different completion requests, invalid packet/tail/frontier rejection,
partial-tail resume, and uncertainty ownership. Multi-queue fixtures check static
signal metadata and token ownership only: no native backend currently implements
that optional SPI, and `StepId` does not identify an execution instance.

The example host has separate fault-injection tests. On Windows, ignored GPU
fixtures execute a four-command plan four times with independent resources,
submitting all executions before waiting and checking every readback:

```powershell
cargo test -p fluxel-rhi --no-default-features --test frame_submission
cargo test -p fluxel-rhi --features vulkan --test execution_plans_gpu -- --ignored --nocapture
```

The latter runs the enabled DX12 and Vulkan routes on an actual GPU. It proves
the tested single-lane plan, explicit dependencies, plan reuse, and End/Steps
completion requests. It does not establish multi-queue conformance, arbitrary
plans, or native partial/indeterminate recovery. The RenderGraph's compiler,
hazard analysis, and resource-retirement tests belong to its own repository.

Failure-boundary tests cover exact None/Partial/Unknown slot cases, clean
presentation rejection/retry, and image/value/checkpoint/uncertainty Drop
ordering through failure drain. The ignored GPU fixtures were explicitly
executed; `#[ignore]` controls routine test selection and is not evidence that
they have never run. The common execution migration was exercised on real GPU
routes; its native visual-capture limits remain separate from the readback
claim.

A reproducible CPU-only validation diagnostic is in
[`tools/benchmarks/execution-validation`](../tools/benchmarks/execution-validation/README.md).
It measures synthetic single-lane plans with and without predecessor waits,
not Execution wrapper overhead, native GL calls, or real RenderGraph workloads.
No validation cache, registration system, or structural optimization follows
from this diagnostic alone.

The additive common execution entry has its own mock tests for immediate native
call timing, partial/uncertain effects, presentation lease retry, cancellation,
and unchanged deferred recovery ownership. Its real-GPU fixture uses the same
four-step Commands plan with explicit same-lane waits and exact buffer readback
on DX12, Vulkan, and WGL; WGL additionally exercises a clear/present sequence.

```powershell
cargo test -p fluxel-rhi --features vulkan,native-gl-wgl --test unified_execution_gpu -- --ignored --nocapture
```

This headless readback/Present smoke does not establish arbitrary cross-lane
synchronization, browser execution, or visual correctness. The optional common
entry does not add native multi-queue or Sparse support. All 15 examples now use the common entry. Their deferred path retains native
pre-recording, including example 13 secondary work, and binds it without a
portable command conversion. GL passes encoding closures directly.

The execution migration was checked on DX12, Vulkan and WGL with focused GPU
readback and presentation smoke. These results do not establish arbitrary
cross-lane synchronization, browser execution, or visual correctness.

On Windows, run a finite native workload with explicit backend selection:

```powershell
$env:VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation'
cargo run -p fluxel-rhi --features examples,vulkan --example 15_multiview -- --backend vulkan --frames 2
$env:FLUXEL_DX12_DEBUG='1'
cargo run -p fluxel-rhi --features examples,dx12 --example 15_multiview -- --backend dx12 --frames 2
```

Some examples need the pinned local assets installed by `tools/examples/fetch-assets.ps1`. The example command proves only the selected route. A successful compile does not prove native execution, and an empty validation log does not prove visual correctness; compare GPU captures with the corresponding C++ reference when accepting scene output.

The DX12 default-resource pool has an opt-in real-adapter churn diagnostic. It
compares 1,024 small buffers and 256 small textures over 4–64 MiB pool pages
against committed allocation. It measures CPU create/drop latency and native
heap-page counts, not GPU frame time or pool-only resident memory. Where DXGI
exposes a segment budget, the diagnostic also prints process-wide usage/budget:

```powershell
cargo +1.87.0 test -p fluxel-rhi-dx12 --features hardware-tests --lib compare_pooled_and_committed_buffer_churn -- --ignored --nocapture
```

The Vulkan default-resource pool has a matching opt-in real-adapter diagnostic.
It compares the full ordinary-resource create/drop path for 1,024 device buffers
and 256 optimal images, reports the retained native pool-page count, and prints
the adapter and driver identifiers. It measures CPU create/drop latency only;
it does not compare against wgpu or measure GPU frame time. When available, it
prints `VK_EXT_memory_budget` process-wide per-heap estimates; these do not
report pool-only resident memory. A device-level pool benchmark is not evidence
of renderer performance:

```powershell
cargo +1.87.0 test -p fluxel-rhi-vulkan --features hardware-tests --lib compare_pooled_and_committed_buffer_texture_churn -- --ignored --nocapture
```

Browser WebGPU and WebGL2 integration tests use Chrome, ChromeDriver, and `wasm-bindgen-test-runner`. Set `WASM_BINDGEN_USE_BROWSER=1` so the runner uses a browser instead of its default Node.js environment. Also set `CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner`, `CHROMEDRIVER`, and `WASM_BINDGEN_TEST_WEBDRIVER_JSON` (with Chrome's `--enable-unsafe-webgpu` argument for WebGPU tests); keep these variables in the same PowerShell session as the test command. See the [wasm-bindgen browser testing guide](https://rustwasm.github.io/docs/wasm-bindgen/wasm-bindgen-test/browsers.html).

```powershell
cargo test -p fluxel-rhi-webgpu --target wasm32-unknown-unknown -- --nocapture
cargo test -p fluxel-rhi --target wasm32-unknown-unknown --features 'webgpu examples' --test webgpu_triangle -- --nocapture
```

The opt-in Review3 WebGPU upload diagnostic emits browser-side recording and
completion time plus staging-page allocations for 1/10/100 encoders and
1/100/1000 small writes. It is intentionally outside ordinary conformance:

```powershell
cargo test -p fluxel-rhi-webgpu --target wasm32-unknown-unknown --features review3-upload-benchmark review3_upload_arena_timing_matrix -- --nocapture
```

Use the same Chrome/WebDriver environment as the preceding WebGPU command.

Run each numbered `webgpu_*` integration test for its scene. These tests present three frames through a real browser GPU device. Keep the corresponding browser screenshot and compare it with local C++ reference captures while excluding the C++ HUD. Examples 05, 13, 14, and 15 have explicit browser visual fallbacks; their tests do not prove the unavailable push-constant, secondary-command, runtime-descriptor-array, or multiview capabilities.

Record the commit, platform, adapter, command, validation result, and observed image or readback for each hardware claim. Classify the result as pass, unsupported, skipped, or failure. Keep portable Core unit-test sources in `crates/core/src/tests/`; keep backend unit-test sources in the owning crate's `src/tests/` tree and native integration checks in the owning crate's root `tests/` directory.

Metal validation additionally requires a macOS SDK link, focused real-GPU readback assertions, and all 15 native examples with inspected screenshots before the backend can be marked complete. An Apple-target cross-check from another host proves type correctness only.

The conditional Review3 Metal Archive Prepare item has an opt-in Apple GPU
diagnostic. It compiles the MSL function before sampling, uses a fresh archive
for each sample, and times only compute archive Prepare/lease setup; it prints
P50/P95/P99 across 51 samples. Run it on a Mac with Metal binary archive
support:

```sh
cargo test -p fluxel-rhi-metal --features hardware-tests --lib metal_archive_prepare_latency_percentiles -- --ignored --nocapture
```

This measures Archive Prepare, not native PSO compilation or GPU execution.

## Android emulator example checks

The Android 14 emulator run covered Vulkan, GLES and WebGL2 examples 01–15,
native readbacks, focused clear-mask GPU regressions, and Android HOME/return
checks. WebGPU adapter acquisition returned null in that emulator's WebView; it
is not a WebGPU pass. Original screenshots and run logs are retained in the
local Temp archive and are not part of the GitHub source tree. Use the native
`tools/examples/android/run-capture.ps1` or browser
`tools/examples/android/browser/run_capture.py` helper for one route at a time.
The browser export stops after three frames for capture; WebGL2 example 13 is
sequential drawing, despite its historical multithreaded-recording filename.
