# Android browser evidence helper

This directory serves one browser probe or one WebGL2/WebGPU wasm example to
an Android emulator. It uses `adb reverse` so the browser loads
`http://localhost`, which is a trustworthy origin for the WebGPU secure-context
requirement. The page POSTs structured capability and completion JSON back to
the local Python server. It also performs a WebGL2 drawing-buffer readback;
success is not inferred from a screenshot alone.

All numbered WebGL2 entries use the shared statically dispatched Core execution
entry. Their encoding closures call GL immediately; the browser helper only
adapts canvas lifecycle, captures, and observable results. It does not cache or
replay RHI commands. The current migration evidence is linked from the
[example guide](../../../../crates/rhi/examples/README.md#verification-status).

The scripts are local developer tools. Generated wasm bundles and screenshots
belong under the ignored local verification archive, not this directory.

## Probe

Start the server in one terminal:

```powershell
python tools/examples/android/browser/browser_runner.py serve --mode probe --result '临时/当前验证/android/2026-10-16/browser/probe.json'
```

Forward the host server to the selected device and open the URL in Android
Browser:

```powershell
adb -s emulator-5554 reverse tcp:8765 tcp:8765
adb -s emulator-5554 shell am force-stop com.android.browser
adb -s emulator-5554 shell am start -a android.intent.action.VIEW -n com.android.browser/.BrowserActivity -d 'http://localhost:8765/?run=probe'
python tools/examples/android/browser/browser_runner.py poll
```

The response records `isSecureContext`, WebGL2 availability and identity, and
whether `navigator.gpu.requestAdapter()` succeeds. Save a device screenshot
separately with `adb exec-out screencap -p > <evidence>.png`.

## One example

The single-example helper builds the bundle, serves a unique URL, matches the
page's run ID, and waits for completion before taking two display samples:

```powershell
python tools/examples/android/browser/run_capture.py 01_triangle --backend webgl2 --evidence target/android-browser-evidence/webgl2
```

It preserves the page result, candidate/source and bundle hashes, and PNGs.
Inspect the images: a successful page readback alone does not prove the Android
compositor has displayed the final canvas. Earlier captures taken immediately
after a POST could still show `loading`; exclude those images from acceptance.
The current helper waits for animation-frame handoff and an additional display
delay. A page timeout or wrong run ID is a failed run, never a backend pass.

For manual use, the separate build and server commands remain available.

Build a web package for exactly one backend/example pair. This command prints
the exported entry function needed below.

```powershell
python tools/examples/android/browser/build_bundle.py 01_triangle --backend webgl2 --out '临时/当前验证/android/2026-10-16/browser/bundles/webgl2/01_triangle'
```

Serve it and then reuse the same `adb reverse` command and Browser activity
command from the probe section:

```powershell
python tools/examples/android/browser/browser_runner.py serve --mode demo --backend webgl2 --demo 01_triangle --entry run_webgl2_triangle --bundle-dir '临时/当前验证/android/2026-10-16/browser/bundles/webgl2/01_triangle' --result '临时/当前验证/android/2026-10-16/browser/webgl2-01_triangle.json'
```

After the page reports `success`, capture two displayed frames with `adb
exec-out screencap -p`. A report with `state: error` includes the JavaScript or
wasm error and must be treated as a failed run even if a canvas is visible.

Use `--backend webgpu` only after the probe reports a usable adapter. The same
workflow then calls the WebGPU export printed by `build_bundle.py`.
