# RHI examples

The 15 numbered examples use the statically typed `fluxel_rhi::api`. Native
examples use the shared `fluxel-host` window lifecycle. Browser examples use
an HTML canvas owned by the caller. DX12, Vulkan, and desktop GL4+ use native
Windows routes; GLES3+ uses EGL; Metal's macOS host path is in progress; WebGPU
and WebGL2 run in a browser. An example is accepted for a backend only
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
| DX12 | Commit `956f277a7ada5ec7decda7a8544e59a9fc2e9ee1` is pushed. All 15 examples ran on a real GPU with the D3D12 debug layer and were visually checked against C++; the complete WGSL manifest regenerates DXIL/SPIR-V byte-for-byte and checks shader ABI. On the current `eeed5bf538ea99e64ecb682f223764495d353799` worktree, all 15 examples exited successfully for two frames with `FLUXEL_DX12_DEBUG=1`. ID 820 in examples 08/12 is a performance hint: D3D12's resource descriptor uses a zero color clear value while those passes clear to other values. D3D12 still clears correctly; this is not a GPU-capability failure. The process-termination live-object summary lacks object names and is not sufficient evidence of a leak. The desktop was locked during the current run, so it adds no visual evidence. | Complete for the pushed visual candidate; current worktree execution regression passed. Revisit optimized clear values only if clear performance warrants a public resource contract change. |
| Vulkan | All 15 examples ran for two frames on a real GPU with `VK_LAYER_KHRONOS_validation`; the pushed visual candidate was checked against C++ with the HUD excluded. The 09 anisotropic sampler path and 14 explicit upload-to-sampling transitions were fixed and rechecked. Example 13 was visually verified from swapchain GPU readback because desktop capture was occluded. Example 07 preserves the reference's intentional 7×7 reflection blur. On the current `eeed5bf538ea99e64ecb682f223764495d353799` worktree, all 15 examples exited successfully for two frames with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`. The backend does not install a Vulkan debug messenger, so this run cannot claim zero validation diagnostics. Desktop was locked, so it adds no visual evidence. | Complete for the pushed visual candidate; current worktree execution regression passed. |
| WebGPU | All 15 browser tests passed on Chrome 155.0.8059.40 using an AMD RDNA 3 adapter (`isFallbackAdapter = false`); each presented three frames at 1280×720. Captures were visually compared with the C++ references with the HUD excluded. The sampler anisotropy path and WASM animation clock were fixed. On the current `eeed5bf538ea99e64ecb682f223764495d353799` worktree, all 15 tests passed again in Chrome 155, presenting three frames each. This rerun did not produce a new screenshot set. | Complete for the pushed visual candidate; current worktree execution regression passed. |
| Metal | The initial backend implementation includes adapter/device/queue creation, WGSL→MSL translation, native shader-library compilation, MTLBuffer allocation, 2D textures/views in a limited RGBA/BGRA/Depth32 plus render-only Depth32FloatStencil8 on Apple2/Mac2-family devices, texture-to-texture blit, buffer-to-texture blit, aligned texture upload and texture-to-buffer copy, completion-gated buffer readback, samplers, scalar bind groups, dynamic uniform offsets, a narrow TriangleList pipeline with Depth32 and combined Depth32FloatStencil8 attachment/pipeline state, including stencil operations and pass load/store, device-queried 2/4/8-sample textures and color resolve, cull/blend/write-mask lowering, and direct plus indirect compute dispatch with declared workgroup sizes and `rgba8unorm` write-only storage textures. Indirect dispatch is enabled only when the device reports the required Metal GPU family; it validates the 12-byte argument range and retains its buffer through GPU completion. Direct command encoding, Commands plus one Present step, CAMetalLayer acquire, and asynchronous marker-based completion are implemented. The AppKit host routes all examples 01–15 to its retained `CAMetalLayer`. Examples 05, 10, 11, 13, 14 and 15 use explicit visual fallbacks for native capabilities Metal does not advertise; these paths do not claim native push constants, indirect draws, occlusion queries, secondary commands, runtime descriptor indexing, or multiview. The opt-in `crates/metal/tests/hardware.rs` fixture checks submitted GPU buffer bytes through completion-gated readback (`cargo test -p fluxel-rhi-metal --features hardware-tests --test hardware`). The fixture type-checks for `aarch64-apple-darwin`; there has been no macOS SDK link, Apple GPU run, Metal example execution, or visual review. | In progress; run the focused fixture on Apple GPU, fix any runtime failures, then complete all 15 examples with inspected screenshots and diagnostics. |
| GL4+ | WGL on AMD Radeon 780M (driver 32.0.21028.2002) passed the concentrated 01–15 real-GPU run and independent backend review. All examples have visually reviewed 1280×720 captures against C++ references with the HUD excluded. Example 11 matches the C++ occlusion-query samples (teapot 246605, sphere 2878). Example 12's readback image matches the C++ dragon outside the HUD: RGB MAE 0.001/255, P95 0/255, 93 differing pixels among 748,800. Examples 13–15 use sequential recording, fixed texture-array, and per-eye stereo fallbacks without claiming unavailable native capabilities. Example 15 matches the C++ reference outside the HUD and window border: RGB MAE 0.001/255, P95 0/255. The current worktree was rebuilt on 2026-10-15 and all 15 WGL examples exited successfully for two frames. This adds runtime evidence, not new visual captures. | Complete for the WGL profile on the tested AMD Radeon 780M; current worktree regression passed execution only. |
| GLES3+ | EGL/GLES 3.1 on Mesa D3D12 over AMD Radeon 780M ran all 15 examples for two frames on the current tree. Opt-in `glReadPixels` captures saved each example's last presented default-framebuffer image before presentation; the PNGs were opened beside the C++ references and the visible scene content matches with the HUD excluded. The corrected GL front-face lowering is included. Compute storage-image output and occlusion-query visibility readback ran on GLES; the native visibility result is boolean, so exact positive sample counts are not portable. | Visual and execution checks pass for the tested Mesa D3D12 GLES 3.1 profile. This does not establish WebGL2 completion. |
| WebGL2 | Examples 01–15 ran for three frames on Chrome 155.0.8059.40; the current full browser suite has 16 test targets and 20 passing tests, including screenshot row-layout fixtures and four Surface/context lifecycle cases. The lifecycle checks verify exclusive leases, resize/reconfiguration, generation changes, rejection of acquire after an external canvas resize, and terminal behavior after host-notified or driver-detected context loss. The full suite passed after correcting GL front-face lowering and the immediate texture-copy path; the 15 browser captures were opened side-by-side with the WebGPU candidates and visually match the expected scenes. Example 07’s dragon and reflection now match after preserving the public `FrontFace` value. Example 08’s single-sample fallback matches the WebGPU composition; WebGL2 still does not claim multisample attachment support. Example 12’s readback fixture performs two partial, vertically offset copies with a stale 1×1 scissor enabled, then verifies upper-left row ordering, exact destination bytes, byte offset, and row padding. Example 06 uses a WGSL raster-filter fallback because WebGL2 lacks compute/storage-texture support. Example 10 uses indexed draws with CPU-owned indirect arguments and keeps `indirect_draw = false`. Example 11 uses an immediate visual fallback and keeps `occlusion = false`; it does not claim query-result support. Examples 13–15 retain sequential/fixed-resource fallbacks and make no native secondary-command, runtime descriptor-indexing, or multiview claims. | Visual review is complete for the tested WebGL2 scenes; native capability gaps remain explicitly unsupported and use documented visual fallbacks. |

The verified DX12 candidate reports only D3D12 message ID 820 for the offscreen color
targets in examples 08 and 12. It is a performance hint about the optional
optimized clear value; the API does not declare a fixed pass clear color, and
the clear still produces the requested image. The rest of the captured DX12
examples produced no D3D12 debug messages.

The local captures, run logs, hardware details, and exact worktree state are
retained in an ignored local verification archive and are not part of a
published backend claim. The reference images come from the local C++ Vulkan
example build; the HUD is excluded from comparison.

### Current full execution regression

On 2026-10-15, all 15 examples were rerun from the directory-reorganized tree
at HEAD `eeed5bf538ea99e64ecb682f223764495d353799`. DX12 and Vulkan each ran
two frames per example; WebGPU presented three frames per browser test. No
DX12, Vulkan, WebGPU, or example source files differed from that HEAD during
these runs. The worktree also contained unrelated, uncommitted Metal crate
work. These results verify execution only; they do not bind a future commit
that changes the tested sources.

| Backend | Device and diagnostics | Result record |
| --- | --- | --- |
| DX12 | AMD Radeon 780M, driver 32.0.21028.2002; `FLUXEL_DX12_DEBUG=1`. All 15 processes exited successfully. The 08/12 ID 820 performance hints remain as described above. Windows was locked during the attempted capture, so that image is invalid. Logs retained in the ignored local verification archive. |
| Vulkan | AMD Radeon 780M, AMD proprietary driver 25.10.28.02; Khronos validation layer selected. All 15 processes exited successfully. The backend does not install a debug messenger, so diagnostic cleanliness is unverified. Windows was locked; no visual evidence was added. Logs retained in the ignored local verification archive. |
| WebGPU | Chrome and ChromeDriver 155.0.8059.40; real AMD adapter, not a fallback adapter. All 15 browser tests passed and presented three frames. This rerun saved no new screenshots. Logs retained in the ignored local verification archive. |

The current worktree run is not a replacement for opening and reviewing new
visual captures. Prior pushed-candidate visual evidence remains recorded in
the backend rows above.

After adding the Apple/Metal host route to examples 01–15, each changed example
was regressed on the already supported backends: DX12 and Vulkan each presented
two frames, and Chrome 155 WebGPU presented three frames (all fifteen focused
browser tests passed). These focused reruns verify execution only and produced
no new visual captures. Exact commands and working-tree diff fingerprints are
recorded in the ignored local verification archive.

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

On Linux, the EGL/GLES immediate examples use the same host lifecycle and
accept `--backend gles`:

```bash
cargo run -p fluxel-rhi --no-default-features --features examples,native-gles-egl --example 01_triangle -- --backend gles --frames 2
```

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

| Example | What it demonstrates |
| --- | --- |
| 01 triangle | Draws an indexed triangle and presents it to the surface. |
| 02 texture | Lights a textured quad using a uniform buffer, a sampled texture, and a sampler. |
| 03 instancing | Renders a planet, a field of instanced rocks, and stars with depth testing. |
| 04 dynamic uniform buffer | Draws a grid of cubes, selecting a different dynamic uniform-buffer range for each draw. |
| 05 push constants | Changes per-draw transforms through push constants where the backend supports them, with a uniform-buffer fallback elsewhere. |
| 06 compute | Runs emboss, edge-detection, and sharpening compute filters, then displays the source and filtered images side by side. |
| 07 offscreen | Renders a dragon and reflective floor into an offscreen target, then samples that image while drawing the final scene. |
| 08 msaa | Renders the Voyager model with multisample anti-aliasing (MSAA) and resolves the result for presentation. |
| 09 texture mipmap | Generates texture mip levels and shows how filtered sampling changes across a moving tunnel. |
| 10 indirect draw | Renders a field of plants using GPU-generated indirect draw commands on capable backends. |
| 11 occlusion query | Uses occlusion queries to determine whether scene objects are visible and reads back the query results. |
| 12 screenshot | Renders a dragon into a texture, copies the rendered image to a readback buffer, and saves a screenshot. |
| 13 multithreaded recording | Compares secondary or parallel command recording where available and uses sequential primary recording as the documented fallback. |
| 14 descriptor indexing | Selects textures through runtime descriptor indexing where available and uses a fixed texture array as the visual fallback. |
| 15 multiview | Renders a stereo scene with native multiview where available and renders each eye in a separate pass as the fallback. |

Follow the public coordinate contract in `documents/design-rhi.md` when
interpreting a vertical flip or front-face difference. A backend must adapt
its native viewport and winding rather than changing shared scene data.
