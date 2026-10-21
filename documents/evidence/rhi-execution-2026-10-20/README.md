# RHI execution contract evidence, 2026-10-20

Candidate: uncommitted worktree based on `a0a68140d986255ac28b28c2ec1032e12c5967eb`.
This packet tests execution of supplied RHI plans. It does not implement a
RenderGraph compiler, hazard analysis, scheduling, or resource retirement.
No example-oriented convenience API was added to Core.

## Results

| Evidence | Result | Scope |
| --- | --- | --- |
| Core unit tests | 20 passed | Static packet/tail/frontier validation, unchanged SPI delivery, plan reuse, signal metadata and injected ownership failures |
| Example host and external SPI tests | 6 passed | Outcome retention through drain, rejection, malformed receipts, cancellation/drain fail-stop without unwinding, consumed native signal access |
| DX12 and Vulkan GPU plan fixtures | 2 passed | Four Commands steps, four independently bound executions per backend submitted before observation, End/Steps requests and exact readback |
| Windows DX12 / Vulkan / WGL examples | 15 per route | Three-frame finite runs, three captures per continuous run, normal window-close teardown |
| Browser WebGPU / WebGL2 examples | 15 per route | Three frames and inspected 1280x720 captures |
| WSL GLES examples | 15 | Three frames and final framebuffer captures on Mesa D3D12 |
| Android Vulkan / GLES | 22 compile checks | Examples 05-15 target compilation; no connected Android device |

The six reachable scene routes total 90 runs. Explicit fallbacks remain
fallbacks. GPU plan fixtures establish only the tested single-graphics-lane
execution, not native multi-queue, partial-resume, or indeterminate recovery.

## Hardware and visuals

Windows: AMD Radeon 780M Graphics, driver `32.0.21028.2002`.
Browser: headed Chrome 155; WebGPU AMD RDNA3 with `isFallbackAdapter = false`;
WebGL2 ANGLE Radeon 780M / D3D11. WSL explicitly selected
`GALLIUM_DRIVER=d3d12`: Microsoft D3D12 / AMD Radeon 780M / GLES 3.1,
Mesa 25.0.7. The default WSL software renderer was excluded.

All six contact sheets in `visual/` were opened and inspected. Expected geometry,
colors, reflection, filters, vegetation, query scene, readback dragon, UFOs,
textured cubes, and stereo composition remain visible. Windows sheets include
three observations per scene; animation phases can differ. Automated checks
reject missing, empty, or near-uniform images. Native reports retain executable
and capture hashes, dimensions, timing, and exit status. Full-resolution
originals remain in the local verification archive; contact sheets do not replace
those originals or establish exact frame numbers.

## Candidate binding

`sources/` preserves build snapshots and final changed Rust source hashes.
Browser guard names were qualified relative to their local modules so the same
examples also compile when embedded by integration tests. All 30 browser cases
were rebuilt and rerun after that name-resolution fix. The earlier browser
evidence remains in a separate local candidate directory.

Example 12 received a formatting-only change after the first native build;
it was rebuilt and rerun on all three native routes. Contact sheets and the
native report select those final Example 12 runs. Browser/GLES snapshots were
checked after their runs. Subsequent Core edits concern optional multi-queue
metadata/bridge, unsupported-clear documentation, and tests, without changing
the example rendering paths. Final GPU plan fixtures use the updated Core.

## Reproduction and limits

```powershell
cargo test -p fluxel-rhi-core
cargo test -p fluxel-rhi --no-default-features --test frame_submission
cargo test -p fluxel-rhi --features vulkan --test execution_plans_gpu -- --ignored --nocapture
```

Full runner scripts, raw captures, browser modules, build/run logs, and Android
compile logs remain in the local verification archive. Workspace compilation,
native/WASM builds, and portable checks completed. Clippy retains pre-existing
warnings; this is not a warning-free release gate. There was no version bump,
tag, publication, or Metal GPU acceptance.

Browser diagnostics include a probe favicon 404 and an HTTP connection reset
at shutdown after successful runs, without a GPU error. Vulkan has no debug
messenger installed by this run, so no zero-validation-diagnostics claim is made.
No new Android runtime or device-loss recovery evidence was collected.
