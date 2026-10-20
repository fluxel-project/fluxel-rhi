# Android emulator verification, 2026-10-20

This packet records the numbered examples on one LDPlayer Android 14 emulator.
It covers the Vulkan, GLES and WebGL2 example routes, not full backend
conformance, arbitrary device-loss recovery, or renderer integration.

| Channel | Result | Evidence |
| --- | --- | --- |
| Vulkan | 01–15 completed at least two frames each; screenshots inspected | [Scenes](vulkan/scenes.png), runtime logs and candidate manifests |
| GLES | 01–15 rerun after the native clear fix; screenshots inspected | [Scenes](gles/scenes.png), runtime logs and candidate manifests |
| WebGL2 | 01–15 completed the three-frame exports; drawing-buffer readback and inspected display samples | [Scenes](webgl2/scenes.png), page results and candidate manifests |
| WebGPU | No adapter: `requestAdapter()` returned null on a secure origin | [Probe](browser_probe.json); no WebGPU example pass claimed |

The emulator reports NX809J, Qualcomm Adreno 750 and GLES `OpenGL ES 3.1 v1`.
Native images are 1920×1080. Vulkan driver properties reported API version
4206592 and driver version 8388961. The browser is Android WebView
146.0.7680.119, reporting `WebGL 2.0 (OpenGL ES 3.0 Chromium)`. Browser canvases
are 1280×720; the inspected display samples include browser chrome and scale
the canvas without changing its aspect ratio. Scene sheets exclude that chrome.

## Candidate identity

Runs used base commit `7a189f92eb3e910b99c2716e349418ab1c04d262` plus the
changes in this commit. Per-example candidate manifests retain changed-source
hashes and APK/library or wasm-bundle hashes. [Source hashes](source_hashes.json)
identify the final Rust sources. Documentation, test diagnostics and capture-page
labels were updated after some runs; these did not change the exercised Rust
runtime paths. [Run ledger](runs.json) records the accepted image hashes and
run folders. It does not substitute a clean base SHA for a modified candidate.

## Fixes and focused checks

- Vulkan 14 now allocates its native runtime descriptor array with explicit
  layout capacity. Core boundary tests passed (7 library tests total), and
  DX12 14 passed its targeted two-frame and visible-window regression.
- Android 15 reuses the original scene: Vulkan native multiview, GLES sequential
  stereo. Vulkan 15 survived two HOME/return cycles; Vulkan 14 and GLES 15 also
  rebuilt resources and completed frames after returning. Surface loss waits
  at most two seconds for an Android lifecycle transition, then fails explicitly.
  The foreground timeout branch was reviewed, not injected on the emulator.
- Native GL pass clears previously inherited disabled depth writes from the
  preceding pipeline. GLES 07 therefore showed a fragmented reflection:
  [before](gles/07_before.png), [after](gles/07_after.png). Clear now enables the
  relevant write masks and invalidates applied pipeline state; subsequent
  pipeline binding restores it. Stencil-only clear uses the signed API.
- The actual WGL readback fixture checks color/depth/stencil clearing and mask
  restoration. Its extra depth-stencil format is operation-probed inside the
  fixture, not advertised as newly supported by public discovery. GL native
  library checks passed: 138 tests, with 2 explicit hardware tests ignored by
  the ordinary command. The clear fixture was run explicitly and passed.
- The browser provider has a separate clear implementation. Its focused depth
  clear/mask readback test passed in Chrome on AMD Radeon 780M through
  ANGLE/D3D11. It does not cover browser color/stencil readback. Android examples
  use the Core browser encoder route, whose clear masks were already set.
- Example 12 native GPU readbacks are retained as lossless PNGs:
  [Vulkan](vulkan/12_gpu_readback.png), [GLES](gles/12_gpu_readback.png). Both are
  1920×1080; the red-subject checks counted 489,724 and 489,723 pixels.

## Interpretation and capture limits

Fallback scenes do not prove unavailable native features. GLES uses dynamic
uniforms for push constants, single-sample 08, sequential 13, fixed texture-array
14 and sequential stereo 15. WebGL2 additionally uses raster filtering for 06,
direct draws for 10 and visibility fallback for 11. Its example 13 performs
sequential drawing; the historical filename does not imply multithreaded
recording. [The capture page](webgl2/13_sequential_capture.png) explicitly states
that it stops after three frames. A static completed page and 0 FPS are expected.

Older first display samples occasionally captured a restored page or `loading`
despite a completed readback. Those samples are excluded. The browser ledger
uses inspected second samples, with new confirmed runs for 13 and 14. The
capture tool now uses a unique run ID, compositor handoff and a display delay.
The first 14 attempt never loaded the demo and stayed on the browser splash;
two subsequent runs completed. Failed local captures remain separate.

Reproduce one native route with `tools/examples/android/run-capture.ps1`, or one
browser route with `tools/examples/android/browser/run_capture.py`. Operate one
channel at a time and inspect saved images independently of process exit codes.

Retained logs normalize line endings and trailing blank lines; original local logs remain unchanged.
