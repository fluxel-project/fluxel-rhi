# RHI testing

The workspace contains Core, DX12, Vulkan, WebGPU, and the typed facade. GL and Metal are not active backends in this checkout.

Run portable checks and build the numbered examples from the workspace root:

```powershell
cargo test -p fluxel-rhi-core
cargo check --workspace
cargo build -p fluxel-rhi --features examples,vulkan --examples
```

On Windows, run a finite native workload with explicit backend selection:

```powershell
$env:VK_INSTANCE_LAYERS='VK_LAYER_KHRONOS_validation'
cargo run -p fluxel-rhi --features examples,vulkan --example 15_multiview -- --backend vulkan --frames 2
$env:FLUXEL_DX12_DEBUG='1'
cargo run -p fluxel-rhi --features examples,dx12 --example 15_multiview -- --backend dx12 --frames 2
```

Some examples need the pinned local assets installed by `crates/rhi/examples/fetch-assets.ps1`. The example command proves only the selected route. A successful compile does not prove native execution, and an empty validation log does not prove visual correctness; compare GPU captures with the corresponding C++ reference when accepting scene output.

Browser WebGPU integration tests use Chrome, ChromeDriver, and `wasm-bindgen-test-runner`. Set `CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner`, `CHROMEDRIVER`, and `WASM_BINDGEN_TEST_WEBDRIVER_JSON` (with Chrome's `--enable-unsafe-webgpu` argument), then run for example:

```powershell
cargo test -p fluxel-rhi-webgpu --target wasm32-unknown-unknown -- --nocapture
cargo test -p fluxel-rhi --target wasm32-unknown-unknown --features 'webgpu examples' --test webgpu_triangle -- --nocapture
```

Run each numbered `webgpu_*` integration test for its scene. These tests present three frames through a real browser GPU device. Keep the corresponding browser screenshot and compare it with `临时/参考效果/cpp/` while excluding the C++ HUD. Examples 05, 13, 14, and 15 have explicit browser visual fallbacks; their tests do not prove the unavailable push-constant, secondary-command, runtime-descriptor-array, or multiview capabilities.

Record the commit, platform, adapter, command, validation result, and observed image or readback for each hardware claim. Classify the result as pass, unsupported, skipped, or failure. Keep portable API tests in `crates/core/src/api/tests.rs`; put native checks beside the backend they exercise.
