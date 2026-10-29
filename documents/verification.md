# RHI verification

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

## Numbered-example coverage

The manifest contains two distinct example groups: the portable/basic scenes
01–15 and the native-advanced scenes 16–23.  The latter deliberately exercise
native ray tracing, mesh shading, sparse resources, GPU-counted drawing, or
multi-queue operations; they are not a second spelling of the basic suite.

The table is a historical recorded-smoke index, not a current-candidate
acceptance matrix. `Pass` means the named historical route completed the
stated real-GPU execution smoke, which can be limited to recorded process
completion/readback logs; it does not assert that the present source revision
was rerun or that the scene passed visual acceptance. `Unsupported`
means the route returns the documented absence of the required RHI capability.
`NotRun` means there is no recorded run for that group. `Failure` is reserved
for a recorded failing route and must not be relabelled as a pass merely
because another configuration succeeds.

### DX12

**Coverage:** Basic 01–15 — Pass; native-advanced 16–23 — Pass.

**Evidence and limits:** Windows pre-commit `examples-layout-2026-10-26`
smoke built and ran 23 entries for two frames. Its source identity is
incomplete; a local evidence index later associated source/binary hashes with
`abc2aa6789d4ef7face323cb07a2a26fba239c0d`. This is neither a rerun of the
present candidate nor a visual result.

### Vulkan

**Coverage:** Basic 01–15 — Pass; native-advanced 16–23 — Pass.

**Evidence and limits:** The same Windows pre-commit smoke ran 23 entries for
two frames on Radeon 780M with `VK_LAYER_KHRONOS_validation`. A later no-layer
three-frame/readback and inspected-capture run exists for example 16. The
pre-commit source identity is incomplete, so this is not a current-candidate
matrix.

### WebGPU

**Coverage:** Basic 01–15 — Pass; native-advanced 16–23 — Unsupported.

**Evidence and limits:** Chrome with WebGPU enabled ran a pre-commit browser
smoke for examples 01–15. The exact source identity is incomplete; a later
local evidence index associates it with
`abc2aa6789d4ef7face323cb07a2a26fba239c0d`. Examples 16–23 require
native-only operations unavailable on this route.

### GL4+ (WGL)

**Coverage:** Basic 01–15 — Pass; native-advanced 16–23 — NotRun.

Historical Windows WGL coverage for 01–15 is described in [the examples
guide](../crates/rhi/examples/README.md#verification-status). No advanced-suite
run is recorded. Do not infer current visual acceptance from the smoke result.

### GLES3+

**Coverage:** Basic 01–15 — Pass; native-advanced 16–23 — NotRun.

Historical Android-emulator coverage for 01–15 is described in [the examples
guide](../crates/rhi/examples/README.md#verification-status). This is not
physical-device or current-candidate acceptance.

### WebGL2

**Coverage:** Basic 01–15 — Pass; native-advanced 16–23 — NotRun.

Historical Android-browser coverage for 01–15 is described in [the examples
guide](../crates/rhi/examples/README.md#verification-status). No advanced-suite
run is recorded.

### Metal

**Coverage:** Basic 01–15 — NotRun; native-advanced 16–23 — NotRun.

Apple-target compile/static review exists, but there is no Apple-GPU execution
evidence.

The table does not erase configuration-specific failures from the evidence
record. In particular, the initial Vulkan no-layer sweep failed in example 16;
the later focused fix and no-layer rerun only establish the historical smoke
entry above. Keep both observations with their commands and hashes when
updating this table.

Exercise Vulkan ray-pipeline creation with validation layers disabled as well
as enabled. Cover stages without specialization overrides and stages with
overrides: layer-enabled success alone can hide a native creation failure.
Verify the example's GPU readback and visible output after pipeline creation.

## Compiled execution plans

Core's execution fixtures consume hand-authored compiled plans; they do not
build a RenderGraph. They check unchanged step/wait/payload delivery, plan reuse
with different completion requests, invalid packet/tail/frontier rejection,
partial-tail resume, and uncertainty ownership. Multi-queue fixtures check static
signal metadata and token ownership only; they do not establish native
multi-queue conformance. DX12, Vulkan, and Metal have native multi-queue routes,
but each route still needs backend-specific device and workload evidence.
`StepId` also does not identify an execution instance, so the caller must retain
the producer/consumer relationship when reusing a compiled plan.

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
entry does not add native multi-queue or Sparse support. All basic examples
01–15 use the common entry. Their deferred path retains native
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

Browser WebGPU and WebGL2 integration tests use Chrome, ChromeDriver, and `wasm-bindgen-test-runner`. Set `WASM_BINDGEN_USE_BROWSER=1` so the runner uses a browser instead of its default Node.js environment. Also set `CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner`, `CHROMEDRIVER`, and `WASM_BINDGEN_TEST_WEBDRIVER_JSON` (with Chrome's `--enable-unsafe-webgpu` argument for WebGPU tests); keep these variables in the same PowerShell session as the test command. See the [wasm-bindgen browser testing guide](https://rustwasm.github.io/docs/wasm-bindgen/wasm-bindgen-test/browsers.html).

```powershell
cargo test -p fluxel-rhi-webgpu --target wasm32-unknown-unknown -- --nocapture
cargo test -p fluxel-rhi --target wasm32-unknown-unknown --features 'webgpu examples' --test webgpu_triangle -- --nocapture
```

Run each numbered `webgpu_*` integration test for its scene. These tests present three frames through a real browser GPU device. Keep the corresponding browser screenshot and compare it with local C++ reference captures while excluding the C++ HUD. Examples 05, 13, 14, and 15 have explicit browser visual fallbacks; their tests do not prove the unavailable push-constant, secondary-command, runtime-descriptor-array, or multiview capabilities.

Record the commit, platform, adapter, command, validation result, and observed image or readback for each hardware claim. Classify the result as pass, unsupported, skipped, or failure. Keep portable Core unit-test sources in `crates/core/src/tests/`; keep backend unit-test sources in the owning crate's `src/tests/` tree and native integration checks in the owning crate's root `tests/` directory.

Metal validation additionally requires a macOS SDK link, focused real-GPU
readback assertions, and inspected screenshots for every example group the
backend claims to support before it can be marked complete. At present both
01–15 and 16–23 are `NotRun` for Metal. An Apple-target cross-check from
another host proves type correctness only.

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

## Capability-specific example coverage

The optional common execution entry is verified separately from native feature
capabilities. Migration checks covered source fingerprints, hardware readbacks
and platform limits. They do not add GL deferred recording, native multi-queue
support, or Metal GPU acceptance.

This matrix covers the portable/basic examples 01–15. The separately recorded
historical smoke coverage for both example groups is in [Numbered-example
coverage](#numbered-example-coverage).

**Typed Core/facade integration.** DX12 and Vulkan: workspace check and all
examples build. WebGPU: WASM check and browser backend tests.

**Surface acquire, graphics submit, present.** DX12: numbered Windows
examples. Vulkan: examples 01–15, two frames each with validation enabled.
WebGPU: numbered three-frame browser tests.

**Texture upload, views, samplers, depth.** Examples 02–09 on DX12, Vulkan,
and WebGPU.

**Compute dispatch.** Example 06 on DX12 and Vulkan; example 06 uses a native
compute pass on WebGPU.

**Texture array and indirect draw.** DX12: example 10. Vulkan: example 10,
including multi-draw. WebGPU: example 10 with a texture array and
`drawIndexedIndirect`; separate fixtures cover GPU-count lowering.

**Occlusion query and readback.** DX12: example 11. Vulkan: example 11,
observed teapot 246604 and sphere 2878 samples. WebGPU: example 11 with native
query resolve and staging readback.

**Image readback.** DX12: example 12. Vulkan: example 12, with PPM and GPU
capture pixel-identical. WebGPU: example 12 with texture-to-buffer copy and
async map.

**Secondary raster work.** DX12: example 13. Vulkan: example 13 with native
secondary command buffers. WebGPU: unsupported; example 13 preserves the scene
using sequential primary recording.

**Runtime descriptor indexing.** DX12: example 14. Vulkan: example 14 with
variable descriptor count and nonuniform image indexing. WebGPU: unsupported;
example 14 uses a fixed 32-layer texture for the same scene.

**Two-view multiview.** DX12: example 15. Vulkan: example 15 with native view
mask `0b11` and `ViewIndex`. WebGPU: unsupported; example 15 renders each eye
in a separate pass.

The DX12 column identifies implemented example paths, not a fresh hardware pass for every row. Vulkan and WebGPU screenshot comparisons use the local C++ reference captures, excluding the C++ HUD. The current Vulkan candidate ran all 15 examples for two frames on AMD Radeon 780M (driver `25.10.28.02`) with `VK_LAYER_KHRONOS_validation`; every example exited successfully with zero validation diagnostics. Visual review found matching scenes across all 15. Example 09's requested anisotropic sampler is now enabled when supported by the device. Example 14 declares the texture upload, shader-read, and return-to-common transitions explicitly. Example 13's presentation was visually checked from swapchain GPU readback because desktop capture was occluded. Example 07 retains the source demo's intentional 7×7 reflection blur. The 14 shared-example transition change was rerun on DX12 with the debug layer and no diagnostics. WebGPU ran all 15 browser tests for three frames each in Chrome 155.0.8059.40 on an AMD RDNA 3 adapter (`isFallbackAdapter = false`) and produced fresh 1280×720 captures for visual comparison. Its 05/13/14/15 fallback scenes prove visible output only; they do not prove the unavailable native capabilities in their row.

## Core-interface gap review

This review is separate from the historical example matrix above. Surface and
new visual acceptance were not repeated while the desktop could be locked.
Raw diagnostics and screenshots remain local.

### DX12

**Result:** Explicit memory, native/external execution, resolve/clear, shader
routes, and runtime arrays completed.

**Limit:** 276 focused/headless checks; the tier-4 sparse-tail path is
unavailable on this GPU.

### Vulkan

**Result:** Exact external boundaries, resolve usages/layout, subgroup, and
enabled precision/storage features completed.

**Limit:** 90 checks with Khronos synchronization validation after targeted
fixes; sparse depth formats are unavailable.

### WebGPU

**Result:** Standalone regional resolve, storage-buffer views, and independent
indirect feature reporting completed.

**Limit:** 36 effective headless browser checks on an AMD hardware adapter.

### Metal

**Result:** Word clear and filtered color blit added; upload and resolve usage
contracts corrected.

**Limit:** Apple-target compile/static review only; no Metal tests, as
requested.

### GL4+

**Result:** Native same-sample color image copy added, including nonzero
multisample-array layers.

**Limit:** Two hardware output fixtures plus ordinary unit tests.

### GLES3+

**Result:** Same-sample MSAA copy explicitly Unsupported; single-sample image
copy and standalone MSAA resolve retained.

**Limit:** LDPlayer passes 2D 4x copy but rejects MSAA-array attachments. Prior
WSL native failures remain unresolved, so complete MSAA-copy support is
disabled pending acceptance.

### WebGL2

**Result:** Existing standalone resolve remains the native MSAA transfer route.

**Limit:** Focused AMD/ANGLE hardware test passes resolve and
MSAA-copy-destination rejection; same-sample image copy has no matching
standard API.

## Updating this matrix

Add a row only after recording a reproducible command and result. Keep source compilation, validation-layer execution, and visual comparison separate. Do not infer unexercised backend capabilities from these examples.
