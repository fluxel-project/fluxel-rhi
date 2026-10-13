# RHI examples

The 15 numbered examples use the statically typed `fluxel_rhi::api`. Native
examples use the shared `fluxel-host` window lifecycle. Browser examples use
an HTML canvas owned by the caller. DX12 and Vulkan are the native Windows
routes; WebGPU runs in a browser. The Vulkan examples 01–15 have been run for
two frames with the Khronos validation layer and GPU-image readback comparison
against the local C++ captures.

From the workspace root on Windows:

```powershell
.\crates\rhi\examples\fetch-assets.ps1
cargo run -p fluxel-rhi --features examples,dx12 --example 01_triangle -- --frames 2
```

Run one of examples 01–15 with the Vulkan feature and explicit backend selection:

```powershell
cargo run -p fluxel-rhi --features examples,vulkan --example 01_triangle -- --backend vulkan --frames 2
```

The native host accepts `--backend dx12|vulkan` and `--frames COUNT` for
examples 01–15. Backend selection happens once at startup; GPU objects and
command recording retain their concrete backend type. The example host and
submission helper live in `common/`. The GPU workload remains in each
numbered example or its small sibling modules.

WebGPU browser tests run the same scene workloads through the browser's real
`GPUDevice`, native command encoders, and canvas presentation. Install the
`wasm32-unknown-unknown` Rust target and `wasm-bindgen-test-runner`, then set
`CHROMEDRIVER` to a compatible ChromeDriver binary and configure the runner to
launch Chrome with WebGPU enabled. A `webdriver.json` for the third variable
can contain `{"goog:chromeOptions":{"args":["--enable-unsafe-webgpu","--disable-gpu-sandbox"]}}`.
For example:

```powershell
$env:CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER = 'wasm-bindgen-test-runner'
$env:CHROMEDRIVER = 'C:\path\to\chromedriver.exe'
$env:WASM_BINDGEN_TEST_WEBDRIVER_JSON = 'C:\path\to\webdriver.json'
cargo test -p fluxel-rhi --target wasm32-unknown-unknown --features 'webgpu examples' --test webgpu_triangle -- --nocapture
```

Each `webgpu_*` integration test presents three frames. The browser route
uses WGSL where the existing SPIR-V shaders cannot be translated faithfully.
WebGPU has no push constants, native parallel secondary command recording,
runtime-sized descriptor arrays, or native multiview. Examples 05, 13, 14,
and 15 use explicitly scoped browser paths that preserve the visible scene
with ordinary uniform bindings, sequential primary recording, fixed texture
layers, and one render pass per eye respectively. Their native capability
flags remain false for those unavailable operations.

Examples 01–15 use checked-in DXIL for DX12 and SPIR-V for Vulkan. The SPIR-V
blobs are built from adjacent HLSL sources with `compile-spirv.ps1`; the
script also validates the result for Vulkan 1.3. `fetch-assets.ps1` obtains
the pinned external models and textures required by some examples. Those
assets remain local; see [assets/README.md](assets/README.md).

| Example | Workload |
| --- | --- |
| 01 triangle | Indexed triangle and presentation |
| 02 texture | Textured, lit quad with uniform and sampler bindings |
| 03 instancing | glTF rock field, planet, stars, depth, and instanced draw |
| 04 dynamic uniform buffer | Cube grid with per-draw dynamic uniform offsets |
| 05 push constants | Push constants |
| 06 compute | Compute dispatch |
| 07 offscreen | Offscreen pass sampled by a display pass |
| 08 msaa | Multisampling and resolve |
| 09 texture mipmap | Mipmap generation and sampling |
| 10 indirect draw | Native indirect drawing |
| 11 occlusion query | GPU occlusion query and readback |
| 12 screenshot | Texture readback |
| 13 multithreaded recording | Secondary/native parallel recording |
| 14 descriptor indexing | Capability-gated runtime descriptor indexing |
| 15 multiview | Capability-gated multiview rasterization |

The visual reference is the local `SaschaWillems/Vulkan` C++ checkout. Compare
actual frames with the reference captures under `临时/参考效果/`; the C++ HUD in
the upper-left corner is outside the rendering comparison. Follow the public
coordinate contract in `documents/design-rhi.md` when interpreting a vertical
flip or front-face difference. A backend must adapt its native viewport and
winding rather than changing the example's shared scene data.
