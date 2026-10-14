# RHI examples

The 15 numbered examples use the statically typed `fluxel_rhi::api`. Native
examples use the shared `fluxel-host` window lifecycle. Browser examples use
an HTML canvas owned by the caller. DX12 and Vulkan are the native Windows
routes; WebGPU runs in a browser. An example is accepted for a backend only
after a real GPU run, saved frames inspected against the C++ reference (apart
from its HUD), and a check of relevant native diagnostics. Static scenes are
judged by their visible result; animated scenes are judged by scene content
and motion rather than exact pixel-by-pixel frame alignment. Compilation or a
clean process exit alone is not visual evidence.

## Verification status

This table distinguishes earlier repository evidence from the current visual
alignment work. It is updated as each backend is completed. The current
working tree is not yet a frozen verification candidate.

| Backend | Current evidence | Remaining gate |
| --- | --- | --- |
| DX12 | All 15 examples have real-GPU captures with the D3D12 debug layer. The complete WGSL manifest (01–15, with a separate 15 display job) regenerates DXIL/SPIR-V byte-for-byte and checks shader ABI. Recent WGSL DX12 captures show the same visible result as C++ across all 15: the 05 sphere ring, 13 UFO field, 14 descriptor-array cubes, and 15 stereo scene were specifically rechecked. 07/09/10 are animated or camera-driven and differ in exact frame state; their scene content and visual effect match. | Run the final 15-example diagnostic and visual pass on the exact candidate SHA, then submit and push DX12 before starting Vulkan. |
| Vulkan | Earlier implementation captures exist, but no current-candidate verification has started. Vulkan work is intentionally gated on the completed and pushed DX12 pass. | Start after DX12 is complete and pushed; validate all 15 against the frozen shared examples. |
| WebGPU | Earlier browser integration tests and examples exist; the recent WGSL and example changes have not received a complete browser visual pass. | Defer until Vulkan is finished. |
| GL4+ / WebGL2 | No complete 15-example visual pass has been accepted. | Defer until WebGPU is finished. |

The current DX12 run reports only D3D12 message ID 820 for the offscreen color
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

Each `webgpu_*` integration test presents three frames. The current RHI WebGPU
path does not expose immediate data, native parallel secondary command
recording, descriptor indexing, or native multiview. Examples 05, 13, 14, and
15 use explicit WGSL/browser paths that preserve the visible scene with
ordinary uniform bindings, sequential primary recording, fixed texture
layers, and one render pass per eye respectively. Their native capability
flags remain false for those unavailable operations. Native 14 and 15 shader
variants are generated from WGSL capability variants; they do not add
HLSL or GLSL source files.

DX12 consumes checked-in DXIL and Vulkan consumes checked-in SPIR-V. WGSL is
the maintained shader source for all numbered examples. The
`wgsl-artifacts/manifest.psd1` manifest covers 01–15, including separate
multiview scene and display jobs. `compile-wgsl.ps1` uses Naga,
SPIRV-Tools, SPIRV-Cross, and DXC with binding and vertex ABI checks, plus
Naga IR lowering for the native push-constant path in 05. The complete
manifest regenerates the checked-in DXIL/SPIR-V artifacts byte-for-byte in
no-write verification. Native artifacts are generated outputs, not additional
maintained shader sources. `fetch-assets.ps1` obtains the pinned external
models and textures required by some examples. Those assets remain local; see
[assets/README.md](assets/README.md).

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

Follow the public coordinate contract in `documents/design-rhi.md` when
interpreting a vertical flip or front-face difference. A backend must adapt
its native viewport and winding rather than changing shared scene data.
