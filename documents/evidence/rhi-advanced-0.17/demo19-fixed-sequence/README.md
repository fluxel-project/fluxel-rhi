# Example 19: matched fixed-step sequence and live presentation

NVIDIA Donut Samples `rt_particles` revision
`d766a7697b67b787714c3f8837fb271583306ab7` was built with the retained
`nvidia-fixed-step.patch`. The patch sets the animation delta to 1/60 second,
stops animation at a selected frame and writes every active particle's slot,
position, velocity, color, radius, age, opacity and rotation. Apply the patch
at the root of that checkout and build `rt_particles` in Release mode. For
each of frames 120, 240 and 360, set `FLUXEL_REFERENCE_FRAME` to the frame
number and `FLUXEL_REFERENCE_STATE` to an ASCII file path, run the executable
with `-dx12`, wait for the state file, and capture the 1000 x 709 client area
after the paused window has displayed. The reference UI reports animation
paused; comparison excludes the UI region.

The RHI example was built on Windows MSVC with AMD Radeon 780M, driver
32.0.21028.2002, from a worktree based on `0917271` and the example changes.
Run it separately with `--backend dx12` and `--backend vulkan`, setting
`FLUXEL_RAY_QUERY_FIXED_REFERENCE=1` and `FLUXEL_RAY_QUERY_STATE_DIR` to the
output directory. Both completed successfully; Vulkan used
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` and emitted no errors. The
shader, source, executable, patch and evidence SHA-256 values are in
`sha256.json`. Logs and full-resolution captures are retained here.

The original and RHI particle counts were 31, 30 and 40, with matching slot
IDs at each sampled frame. The maximum absolute field differences were
5.0e-9, 1.19e-7 and 6.0e-8. `comparison.json` gives the image comparison:
original versus DX12 mean RGB error for visible region (400, 0, 1000, 709)
was 4.25, 4.06 and 4.87; DX12 versus Vulkan was less than 0.0004 each time.
The third original capture was opened: it contains the expected geometry, smoke
and paused UI. These numbers do not establish exact image reproduction.

With `FLUXEL_RAY_QUERY_PRESENT=1`, both backends were separately run in a
window and captured at 4, 7 and 10 seconds after launch. The three
`live-{backend}-*.png` images show changing smoke; successive smoke-region
(400, 300, 650, 650) mean RGB changes were 8.14 and 7.98 for DX12, 7.53 and
7.79 for Vulkan. The third DX12 image was opened and inspected. These earlier live frames use
actual frame deltas and rebuild GPU resources every frame. The later
`../demo19-live-reuse/` evidence covers the resource-reusing implementation. Low frame rate changes
the per-frame emission rate and the resulting animation, so this does not yet
reproduce the original real-time behavior or its UI and intersection paths.
