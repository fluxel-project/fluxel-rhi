# RHI examples

Examples 01–23 use the statically typed `fluxel_rhi::api`. Native
examples use the shared `fluxel-host` window lifecycle. Browser examples use
an HTML canvas owned by the caller. DX12, Vulkan, and desktop GL4+ use native
Windows routes; GLES3+ uses EGL; Metal's macOS host path is in progress; WebGPU
and WebGL2 run in a browser. An example is accepted for a backend only
after a real GPU run, saved frames inspected against the C++ reference (apart
from its HUD), and a check of relevant native diagnostics. Static scenes are
judged by their visible result; animated scenes are judged by scene content
and motion rather than exact pixel-by-pixel frame alignment. Compilation or a
clean process exit alone is not visual evidence.

## Execution contract

These examples exercise the RenderGraph-facing RHI contract. Explicit plans,
steps, lanes, waits, and completion facts are intentional. The shared example
host supplies a fixed test workload; its helpers are not public Core APIs.

All 15 examples now use Core's statically dispatched common execution entry.
The shared `execute_frame` helper invokes each GL encoding closure immediately;
its plan contains an explicit same-lane Commands -> Present dependency. Deferred
submission binds already recorded native work through `recorded_commands` and
finishes the same plan shape. Example 13 keeps its native parallel secondary
recording, while GL keeps the documented sequential fallback. Capability
fallbacks and each scene's shader/command generation are unchanged.

The common entry does not introduce a CPU command cache, replay, scheduling,
normal-path idle wait, or per-step fence. `present` hands off an immediate image
now but only binds a deferred image until finish; its final acceptance is read
from the owning execution outcome. Encoding failure still finishes the execution
and retains that outcome through the standalone host's failure drain. Borrowed
`facts()` observations do not replace the owning retry/uncertainty payloads.

Deferred recording does not publish persistent upload or layout state. Pending
state commits are applied only after the host confirms successful execution.
Partial and indeterminate outcomes retain their original acceptance evidence
until issued work is drained; the standalone host then terminates the failed
frame rather than retrying or reusing speculative state. Attachment resources,
secondary work, and readback tickets remain owned through completion.

Deferred frame submission deliberately drains each frame. Immediate workloads
request completion where readback or per-frame resource reuse requires it;
host teardown and browser entry-point exit also require actual completion. A failed drain is not a safe
release condition: native examples abort, and a browser cancellation or failed
drain traps without unwinding the in-flight scene. Discard/reload that failed
WASM instance instead of continuing its invocation. This is an example host
failure policy, not RHI scheduling, device recovery, or renderer retirement.

## Verification status

The common execution migration was verified on a worktree based on `efc4dbb`.
Compilation alone does not establish runtime or visual conformance.

- **Windows DX12 / Vulkan / WGL:** All 15 examples per route completed three
  frames; shared execution GPU fixtures checked exact buffer readback. New
  desktop visual captures were blocked by the Windows lock screen; invalid
  lock-screen images are not acceptance evidence.
- **Desktop browser WebGPU / WebGL2:** All 15 examples per route completed
  three frames with inspected screenshots on the AMD hardware adapter.
  Capability fallbacks remain explicit.
- **WSL GLES:** All 15 completed three frames with inspected framebuffer
  captures on Mesa D3D12 / AMD Radeon 780M. This is the tested GLES driver
  route, not every EGL implementation.
- **Android emulator Vulkan / GLES:** All 15 per route completed two frames
  with inspected screenshots; example 12 completed GPU readback. The
  emulator-reported Adreno 750 is not physical-device acceptance.
- **Android browser WebGL2:** All 15 completed three frames with nonempty
  drawing-buffer readback and inspected screenshots. WebView 146 has no
  available WebGPU adapter.
- **Metal:** Backend selects deferred execution and source integration is
  maintained. Apple GPU and visual acceptance remain unavailable.

Earlier visual baselines and platform-specific capability checks are
summarized in [the verification guide](../../../documents/verification.md). Historical
results are not relabeled as migration-candidate visual verification.
These historical results cover examples 01–15. Examples 16–23 have separate
DX12 and Vulkan real-GPU evidence for the current development worktree;
their final candidate evidence is tracked separately. The Vulkan backend did
not install a debug messenger in the historical migration runs, so no
validation-cleanliness claim is made for those runs. Existing DX12
optimized-clear performance hints do not establish incorrect output or new
native capabilities.

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
the maintained shader source for examples 01–15. The
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
[assets/README.md](assets/README.md). Examples 16–23 exercise native advanced
features absent from standard WGSL, including ray tracing and mesh shaders.
Their maintained native shader sources and generated artifacts are scoped to
those advanced examples; this does not change the WGSL baseline for 01–15.

### Basic examples (01–15)

1. **01 triangle** — Draws an indexed triangle and presents it to the surface.
2. **02 texture** — Lights a textured quad using a uniform buffer, a sampled
   texture, and a sampler.
3. **03 instancing** — Renders a planet, a field of instanced rocks, and stars
   with depth testing.
4. **04 dynamic uniform buffer** — Draws a grid of cubes, selecting a different
   dynamic uniform-buffer range for each draw.
5. **05 push constants** — Changes per-draw transforms through push constants
   where supported, with a uniform-buffer fallback elsewhere.
6. **06 compute** — Runs emboss, edge-detection, and sharpening compute filters,
   then displays the source and filtered images side by side.
7. **07 offscreen** — Renders a dragon and reflective floor into an offscreen
   target, then samples that image while drawing the final scene.
8. **08 msaa** — Renders the Voyager model with multisample anti-aliasing
   (MSAA) and resolves the result for presentation.
9. **09 texture mipmap** — Generates texture mip levels and shows how filtered
   sampling changes across a moving tunnel.
10. **10 indirect draw** — Renders a field of plants using GPU-generated
    indirect draw commands on capable backends.
11. **11 occlusion query** — Uses occlusion queries to determine whether scene
    objects are visible and reads back the results.
12. **12 screenshot** — Renders a dragon into a texture, copies the image to a
    readback buffer, and saves a screenshot.
13. **13 multithreaded recording** — Compares secondary or parallel recording
    where available and uses sequential primary recording as the documented
    fallback.
14. **14 descriptor indexing** — Selects textures through runtime descriptor
    indexing where available and uses a fixed texture array as the visual
    fallback.
15. **15 multiview** — Renders a stereo scene with native multiview where
    available and renders each eye in a separate pass as the fallback.

### Native-advanced examples (16–23)

16. **16 raytracing hello** — Builds acceleration structures and traces
    Microsoft's screen-space triangle on DX12 and Vulkan.
    `FLUXEL_RAY_READBACK` saves an RGB PPM.
17. **17 GPU queries** — Times and counts vertices and primitives in
    Microsoft's HelloTriangle scene on DX12 and Vulkan.
    `FLUXEL_QUERY_READBACK` saves an RGB PPM.
18. **18 meshlet instancing** — Renders Microsoft's ToyRobot meshlet scene
    with mesh shaders on DX12 and Vulkan. `FLUXEL_MESH_READBACK` saves an RGB
    PPM.
19. **19 ray query particles** — Adapts NVIDIA Donut Samples' `rt_particles`
    scene for inline ray queries on DX12 and Vulkan. Live animation uses
    elapsed frame time and reusable GPU resources; fixed-step readbacks can be
    compared with the instrumented original. Ray intersections and shading
    remain approximations.
20. **20 GPU draw count** — Uses compute culling and a GPU-generated count to
    draw the fixed 180th frame of Microsoft's ExecuteIndirect scene on DX12 and
    Vulkan. `FLUXEL_DRAW_COUNT_READBACK` saves an RGB PPM.
21. **21 sparse texture** — Cycles all nine checkerboard mips from Microsoft's
    ReservedResources sample using separate standard-mip heaps and a shared
    packed tail on DX12 and Vulkan. `FLUXEL_SPARSE_MIP` fixes the selected mip;
    `FLUXEL_SPARSE_READBACK` saves an RGB PPM.
22. **22 clear texture** — Clears a texture before rendering and verifies GPU
    readback on DX12 and Vulkan.
23. **23 multi queue** — Simulates and draws 10,000 particles on separate
    DX12 and Vulkan compute and graphics queues with a GPU wait.
    `FLUXEL_NBODY_READBACK` saves the first frame as an RGB PPM.

Follow the public coordinate contract in [the workspace architecture
document](../DESIGN.md#shared-design-rules) when interpreting a vertical flip
or front-face difference. A backend must adapt its native viewport and winding
rather than changing shared scene data.
