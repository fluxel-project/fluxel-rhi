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

- Vulkan and GLES examples 01–13 have captured and visually inspected Android
  output. Example 12 also produced inspected GPU readback images on both routes.
- GLES example 14 has captured and inspected output using the existing fixed
  texture-array fallback.
- Vulkan example 14 exposed a native runtime-array descriptor allocation failure.
  The explicit `RuntimeSized { max_count }` layout fix compiles, but its Android
  rerun was interrupted by the emulator disconnecting before APK installation.
  The fix is not yet verified on the emulator; the affected DX12 path also
  still needs its targeted runtime regression.
- Example 15 is not integrated into this runner yet. Browser WebGL2/WebGPU
  helpers are in `../android/browser/`; no Android browser runtime result is
  claimed by this checkpoint.

These are partial verification results, not an Android conformance claim.
