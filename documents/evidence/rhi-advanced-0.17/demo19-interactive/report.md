# Example 19: interactive panel

Built on Windows MSVC with AMD Radeon 780M driver 32.0.21028.2002; the
source, executable, dependency lock and shader hashes are in `sha256.json`.
The host git dependency is `b310f3fac4b8319343ef13341cd284344343c2ac`.
This packet was captured before the DX12 and GL manifests were unified onto
that same host revision; its lockfile hash represents that earlier candidate.
The later unified-dependency replay is recorded separately.

DX12 debug and Vulkan validation each rendered fixed frames 120/240/360 at
1000 x 709, exited successfully and emitted no diagnostics. Comparing
(400, 0, 1000, 709) to the stride-24 NVIDIA reference gives DX12 mean RGB
errors 0.877/0.940/0.846; Vulkan is within 0.00003. The panel is disabled
in this headless mode.

The `input-probe.py` harness launched each backend in live mode, sent mouse
events to pause, then selected beam and logo, basis and primary reorientation,
and accumulated-vector orientation, secondary geometry, and eight MLAB
fragments. Captures and logs are retained; the two paused images match
pixel for pixel in the UI-free rectangle. Both processes remained alive
and were terminated by the harness. The DX12 final panel screenshot was
opened and inspected.

This is a functional bitmap panel with click-to-cycle choices and drag-based
emitter values. It does not reproduce ImGui's exact appearance, dropdown
interaction, text editing or the reference camera controls. Fixed-step
images and these probes do not prove every control mode matches NVIDIA.
