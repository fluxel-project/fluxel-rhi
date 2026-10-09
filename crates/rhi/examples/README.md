# RHI examples

This directory contains the numbered ports selected from the local
`SaschaWillems/Vulkan` C++ checkout. Each demo uses the shared `fluxel-host`
lifecycle framework and portable RHI calls. Platform lifecycle and backend
composition live under `common/`, grouped by operating system in `windows.rs`,
`android.rs`, `web.rs`, and `apple.rs`.

Build and run it from the workspace root:

```powershell
.\crates\rhi\examples\fetch-assets.ps1
cargo check -p fluxel-rhi --features examples --example 01_triangle
cargo run -p fluxel-rhi --features examples --example 01_triangle -- --backend vulkan
```

The fetcher downloads the 20 pinned source assets used by the numbered ports
and checks their SHA-256 digests. Their binaries are local build inputs and are
excluded from version control; see [assets/README.md](assets/README.md) for
provenance and attribution. `--frames COUNT` closes a native example after the
given number of rendered frames, which is useful for the later validation pass.

On Windows, the DX12, Vulkan, and GL adapters are implemented in
`common/windows.rs`. The new examples and backend support have not yet been
compiled or run on these backends; see the root `进度.md` for the current state.
On Android, the Vulkan NativeActivity can be built, packaged, installed, and
launched on an attached x86_64 emulator with:

```powershell
.\crates\rhi\examples\android\run.ps1 -Serial emulator-5554
```

The Android adapter and NativeActivity entry point are in
`common/android.rs` and `common/mod.rs`; the package manifest and runner are in
`crates/rhi/examples/android/`. The verified Android run used the Vulkan 1.3 emulator and
presented the reference triangle.

The framework parses `--backend`, `--width`, `--height`, and `--frames`, then routes
`init`, `update`, `render`, `resize`, device-loss/recovery, and `close` through
the `fluxel-host` callbacks. Platform window handles and Vulkan objects stay in
the framework adapter. `01_triangle.rs` contains one demo implementation and
does not call Vulkan APIs directly.

Examples author shaders in WGSL. RHI preserves device-native direct inputs and
uses its optional `naga` feature for source/target pairs Naga supports. The
feature is enabled by default. If the feature is disabled or the requested
conversion is unsupported, shader creation returns a structured
`Unsupported` error. The public API still accepts any code form the selected
device consumes natively; WGSL is the examples' source choice, not an RHI input
restriction.

## SaschaWillems/Vulkan porting plan

The initial porting set contains **15 selected examples** from
`SaschaWillems/Vulkan`. The 15 ports are written through the portable RHI API
and shared example framework. Their Windows DX12, Vulkan, and GL validation
pass has not started yet.

| Example | Main capability to verify | Group |
| --- | --- | --- |
| `triangle` | Graphics pipeline and draw | Baseline |
| `texture` | Texture, sampler, and resource binding | Baseline |
| `instancing` | Instanced draw | Baseline |
| `dynamic_uniform_buffer` | Uniform buffers and dynamic updates | Baseline |
| `push_constants` | Push constants | Baseline |
| `compute` | Compute pipeline and dispatch | Baseline |
| `offscreen` | Offscreen rendering and sampling the result in a mirror pass | Baseline |
| `msaa` | Multisampling and resolve | Baseline |
| `texture_mipmap` | Mipmap generation and use | Baseline |
| `indirect_draw` | Indirect draw | Baseline |
| `occlusion_query` | GPU occlusion queries | Baseline |
| `screenshot` | Texture-to-buffer copy and CPU readback | Baseline |
| `multithreaded_recording` | Parallel command recording | Baseline |
| `descriptor_indexing` | Descriptor indexing | Capability-gated |
| `multiview` | Multiview rendering | Capability-gated |

The core validation path is:

```text
triangle -> texture -> offscreen -> compute -> indirect_draw -> occlusion_query -> multithreaded_recording
```

`descriptor_indexing` and `multiview` must query the device's published
capabilities and refuse unsupported routes explicitly. Vulkan-specific
`dynamicrendering`, `timelinesemaphore`, subpass and input-attachment examples,
and advanced mesh-shader and ray-tracing examples are outside this initial set.

## Migration record: 01 triangle

Reference checkout: `E:\moyy\program\cpp\Vulkan`, revision
`41a4410243fca7640a2dd0115a5ff5cb9a29494`.

Primary source: `examples/triangle/triangle.cpp`. Shader sources:
`shaders/glsl/triangle/triangle.vert` and
`shaders/glsl/triangle/triangle.frag`. The C++ `base/` supplies the window,
surface/swapchain, depth format, render pass, framebuffers, and camera. The
triangle class owns geometry, uniforms, descriptors, pipeline, and commands.

| C++ behavior | RHI port |
| --- | --- |
| Three position/color vertices and indices `[0, 1, 2]`; staging copies into device buffers | Preserves the same 24-byte vertex records, index order, upload jobs, and indexed draw. |
| Two per-frame uniform blocks containing projection, model, and view matrices | Uses two UBOs and corresponding bind groups; waits for a slot's completion before reusing it. |
| GLSL vertex/fragment shaders; triangle list; two `vec3<f32>` attributes; depth test/write `LessEqual`; one sample; dynamic viewport/scissor | Authors equivalent WGSL and creates the shader modules through `Device::create_shader`; the pipeline and vertex layout use portable RHI descriptors. |
| Swapchain color clear/store; depth clear to 1 and discard | Acquires a presentation frame, clears color to `[0, 0, 0.2, 1]`, clears depth to 1, and stores the color result. |
| Per-frame indexed draw and present | Records `draw_indexed(0..3)`, submits with `present_after`, and waits for presentation completion. |
| Camera at `(0, 0, -2.5)`, perspective 60°, near/far `1/256`, identity model | Reproduces the camera matrices and recomputes projection aspect from the drawable extent. |
| Resize and teardown callbacks in `base/` | `fluxel-host` owns resize and surface lifecycle; the framework reconfigures presentation before the demo recreates its depth attachment. |

Observable reference output: window title `Basic indexed triangle`; a centered
triangle with red `(1, 1, 0)`, green `(-1, 1, 0)`, and blue `(0, -1, 0)`
vertices on a dark blue clear background. The example has no triangle-specific
animation.

The numbered examples are registered in `crates/rhi/Cargo.toml` and gated by
the `examples` feature. The migration record above documents the triangle in
detail; the other ports should be compared with their corresponding C++ source
when the Windows backend validation pass begins.
