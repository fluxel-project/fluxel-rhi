# RHI examples

The 15 numbered examples use the statically typed `fluxel_rhi::api`. Native
examples use the shared `fluxel-host` window lifecycle. Browser examples use
an HTML canvas owned by the caller. DX12, Vulkan, and desktop GL4+ are native
Windows routes; WebGPU runs in a browser. An example is accepted for a backend only
after a real GPU run, saved frames inspected against the C++ reference (apart
from its HUD), and a check of relevant native diagnostics. Static scenes are
judged by their visible result; animated scenes are judged by scene content
and motion rather than exact pixel-by-pixel frame alignment. Compilation or a
clean process exit alone is not visual evidence.

## Verification status

This table distinguishes earlier repository evidence from the current visual
alignment work. It is updated as each backend is completed. The DX12 and
Vulkan rows refer to their committed verification candidates; later shared
example changes require rerunning affected backend paths.

| Backend | Current evidence | Remaining gate |
| --- | --- | --- |
| DX12 | Commit `956f277a7ada5ec7decda7a8544e59a9fc2e9ee1` is pushed. All 15 examples ran on a real GPU with the D3D12 debug layer and were visually checked against C++; the complete WGSL manifest regenerates DXIL/SPIR-V byte-for-byte and checks shader ABI. The 08/12 ID 820 messages are clear-value performance hints and do not affect output. | Complete. Rerun only paths affected by a later shared change. |
| Vulkan | All 15 examples ran for two frames on a real GPU with `VK_LAYER_KHRONOS_validation`; each produced zero validation diagnostics and was visually checked against C++ with the HUD excluded. The 09 anisotropic sampler path and 14 explicit upload-to-sampling transitions were fixed and rechecked. Example 13 was visually verified from swapchain GPU readback because the desktop capture was occluded. Example 07 preserves the reference's intentional 7×7 reflection blur. | Complete for this candidate. Continue to WebGPU after the Vulkan changes are committed and pushed. |
| WebGPU | All 15 browser tests passed on Chrome 155.0.8059.40 using an AMD RDNA 3 adapter (`isFallbackAdapter = false`); each presented three frames at 1280×720. Fresh captures were visually compared with the C++ references with the HUD excluded. The sampler anisotropy path and WASM animation clock were fixed. | Complete for this candidate. Continue to GL after this backend commit is pushed. |
| GL4+ / WebGL2 | WGL on AMD Radeon 780M (driver 32.0.21028.2002) passed the concentrated 01–15 real-GPU run and independent backend review. All examples have visually reviewed 1280×720 captures against C++ references with the HUD excluded. Example 11 matches the C++ occlusion-query samples (teapot 246605, sphere 2878). Example 12's readback image matches the C++ dragon outside the HUD: RGB MAE 0.001/255, P95 0/255, 93 differing pixels among 748,800. Examples 13–15 use sequential recording, fixed texture-array, and per-eye stereo fallbacks without claiming unavailable native capabilities. Example 15 matches the C++ reference outside the HUD and window border: RGB MAE 0.001/255, P95 0/255. | GL4+ is complete for the WGL profile on the tested AMD Radeon 780M. This does not establish GLES3+ or WebGL2 completion. WebGL2 has compile-only evidence for 09–12 and no visual pass; 09 blit, 10 indirect draw, 11 query/readback, and 12 buffer readback are not claimed as supported there. |

The verified DX12 candidate reports only D3D12 message ID 820 for the offscreen color
targets in examples 08 and 12. It is a performance hint about the optional
optimized clear value; the API does not declare a fixed pass clear color, and
the clear still produces the requested image. The rest of the captured DX12
examples produced no D3D12 debug messages.

The local captures, run logs, hardware details, and exact worktree state are
recorded under the ignored `临时/` directory during development. These local
files are not part of a published backend claim. The reference images come
from the local C++ Vulkan example build; the HUD is excluded from comparison.

From the workspace root on Windows:

```powershell
.\tools\examples\fetch-assets.ps1
cargo run -p fluxel-rhi --features examples,dx12 --example 01_triangle -- --frames 2
```

Run one of examples 01–15 with the Vulkan feature and explicit backend selection:

```powershell
cargo run -p fluxel-rhi --features examples,vulkan --example 01_triangle -- --backend vulkan --frames 2
```

The native deferred host accepts `--backend dx12|vulkan` and `--frames COUNT`
for examples 01–15. WGL immediate routes accept `--backend gl` for the
examples that expose that path. Backend selection happens once at startup; GPU objects and
command recording retain their concrete backend type. The example host and
submission helper live in `utils/`. Each GPU workload and its private modules
live within that example's numbered directory.

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

Each `webgpu_*` integration test presents three frames. The current RHI WebGPU
path does not expose immediate data, native parallel secondary command
recording, descriptor indexing, or native multiview. Examples 05, 13, 14, and
15 use explicit WGSL fallback paths that preserve the visible scene with
ordinary uniform bindings, sequential primary recording, fixed texture
layers, and one render pass per eye respectively. GL 13–15 use sequential,
fixed-resource fallbacks through immediate encoding. Capability flags remain
false for unavailable native operations. Native 14 and 15 shader variants are
generated from WGSL capability variants; they do not add HLSL or GLSL source
files.

DX12 consumes checked-in DXIL and Vulkan consumes checked-in SPIR-V. WGSL is
the maintained shader source for all numbered examples. The
`tools/examples/wgsl-artifacts/manifest.psd1` manifest covers 01–15, including
separate multiview scene and display jobs. `tools/examples/compile-wgsl.ps1` uses Naga,
SPIRV-Tools, SPIRV-Cross, and DXC with binding and vertex ABI checks, plus
Naga IR lowering for the native push-constant path in 05. The complete
manifest regenerates the checked-in DXIL/SPIR-V artifacts byte-for-byte in
no-write verification. Native artifacts are generated outputs, not additional
maintained shader sources. Example 15 has a native multiview display WGSL
source and a sequential per-eye fallback WGSL source; WebGPU and GL use the
fallback. `tools/examples/fetch-assets.ps1` obtains the pinned external
models and textures required by some examples. Those assets remain local; see
[assets/README.md](assets/README.md).

| Example | Workload |
| --- | --- |
| 01 triangle | Indexed triangle with surface presentation |
| 02 texture | Textured and lit quad with uniform and sampler bindings |
| 03 instancing | glTF rock field, planet, stars, depth testing, and instanced draws |
| 04 dynamic uniform buffer | Cube grid with a distinct dynamic uniform offset for each draw |
| 05 push constants | Per-draw push constants on backends that expose them |
| 06 compute | Compute dispatch followed by display of the generated texture |
| 07 offscreen | Offscreen scene pass sampled by a display pass |
| 08 msaa | Multisample anti-aliasing (MSAA) with resolve into a presentation texture |
| 09 texture mipmap | Mipmap generation and filtered sampling across mip levels |
| 10 indirect draw | Native GPU indirect drawing commands |
| 11 occlusion query | GPU occlusion queries with result readback |
| 12 screenshot | Render-target texture readback into a screenshot file |
| 13 multithreaded recording | Secondary or native parallel command recording, with a documented sequential fallback |
| 14 descriptor indexing | Capability-gated runtime indexing into a texture descriptor array, with a fixed texture-array visual fallback |
| 15 multiview | Capability-gated native multiview rendering with sequential per-eye visual fallbacks |

Follow the public coordinate contract in `documents/design-rhi.md` when
interpreting a vertical flip or front-face difference. A backend must adapt
its native viewport and winding rather than changing shared scene data.
