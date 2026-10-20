# Android native example capture runner

This standalone Cargo workspace builds a NativeActivity APK for one shared RHI
example at a time. It uses Vulkan deferred commands or GLES immediate commands
on the Android window owner thread. The numbered example owns its scene and
WGSL-derived shaders; this runner owns window adoption, presentation, completion,
and capture plumbing.

From the repository root on Windows, with the Android SDK, NDK, Rust Android
target, `adb`, and a debug signing keystore available:

```powershell
./tools/examples/android/run-capture.ps1 -Backend vulkan -Demo 01_triangle -Evidence target/android-evidence/vulkan
./tools/examples/android/run-capture.ps1 -Backend gles -Demo 12_screenshot -Evidence target/android-evidence/gles
```

The helper targets `x86_64-linux-android` and defaults to `emulator-5554`.
It requires two completed-frame markers, saves two sampled Android screenshots
and the runtime log, and pulls the completed GPU readback PPM for example 12.
The screenshots include Android system UI. Builds and APKs remain in ignored
target directories. Fetch example assets with `tools/examples/fetch-assets.ps1`
before building scenes that use external models or textures.

## Verification checkpoint

- Vulkan examples 01–15 completed two frames each on the Android 14 emulator
  reporting Adreno 750 at 1920×1080. Saved screenshots were inspected against
  the C++ references; example 12 also produced a checked GPU readback image.
- Vulkan example 14 now runs with the explicit `RuntimeSized { max_count }`
  capacity and native descriptor indexing. Core capacity boundary tests and
  the affected DX12 example's targeted runtime regression passed.
- Example 15 uses native Vulkan multiview. Two HOME/return cycles rebuilt the
  Vulkan device, surface and scene and completed two frames after each return;
  example 14 also passed a HOME/return check. Surface loss waits briefly for an
  Android lifecycle transition, then reports an error if none arrives. This is
  bounded failure, not a claim of arbitrary surface-loss recovery.
- GLES examples 01–15 completed two frames each with inspected screenshots;
  example 12 also produced checked GPU readback. Example 15's sequential stereo
  route passed HOME/return with resource rebuilding. Existing fallbacks remain
  explicit: dynamic uniforms for push constants, single-sample example 08,
  sequential example 13, fixed texture-array example 14, and stereo example 15.
- GLES example 07 exposed stale depth write masks suppressing the next pass's
  depth clear. Native GL pass clears now enable attachment write masks and
  invalidate cached pipeline state before clearing; stencil-only clear uses the
  signed API. Android GLES 01–15 and desktop GL 07 were rerun, and a WGL GPU
  readback regression checks color/depth/stencil clears and pipeline mask
  restoration. The browser implementation has a separate clear path.
- Browser helpers are in `../android/browser/`. WebGL2 examples 01–15 completed
  three-frame exports with checked drawing-buffer readback and inspected display
  samples. WebGPU adapter acquisition returned null, so no WebGPU example run
  is claimed. The browser page explicitly labels sequential drawing for 13 and
  stops after the requested capture frames.

The retained [Android evidence packet](../../../documents/evidence/android-2026-10-20/README.md)
records the tested paths, candidate hashes, images and remaining limits.

The helper writes per-example candidate metadata with HEAD, changed source
hashes and APK/library hashes, and archives runtime logs before checking for
errors. Evidence from a modified tree must retain that metadata alongside the
images; a HEAD SHA alone does not identify the tested candidate.

These are partial verification results, not an Android conformance claim.
