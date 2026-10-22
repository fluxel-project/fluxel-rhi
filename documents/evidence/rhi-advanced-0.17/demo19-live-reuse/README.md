# Example 19: reusable live scene

The live renderer keeps scene buffers, acceleration structures, bindings,
pipeline, output texture and display pipeline across frames. Each frame it
advances the particle simulation with elapsed time, uploads the particle data,
traces and presents. The source sample also uses actual frame deltas and emits
at most one particle per update. Results therefore depend on the runtime frame
rate; the fixed-step reference experiment is in `../demo19-fixed-sequence/`.

The captures were made on Windows MSVC with AMD Radeon 780M, driver
32.0.21028.2002, from a worktree based on `0917271` with the example changes.
`sha256.json` records source, compiled shader, executable and evidence hashes.
Build with `cargo build -p fluxel-rhi --features examples,dx12,vulkan --example
19_ray_query_particles`. Set `FLUXEL_RAY_QUERY_PRESENT=1` and
`FLUXEL_RAY_QUERY_TRACE=1`, then run each backend separately with
`--backend dx12 --frames 120` and `--backend vulkan --frames 120`. DX12 used
`FLUXEL_DX12_DEBUG=1`; Vulkan used
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`. Both exited successfully with
no debug or validation diagnostics. The trace reports frame 120 at 1.023 s
with 19 active particles for DX12 and 1.896 s with 34 active particles for
Vulkan. These are observations on this device, not performance benchmarks.

The same binary was separately run in each backend's window and captured at
4, 7 and 10 seconds after launch. `motion.json` gives consecutive RGB mean
absolute differences within smoke region (400, 300, 650, 650): 10.89 and 9.40
for DX12, 10.32 and 9.53 for Vulkan. The third DX12 image was opened and shows
colored smoke and reflected geometry. Capture logs record window size and
frame traces. Differences in frame cadence mean the two live sequences are not
expected to contain the same particles at the same wall-clock time. Ray
intersection, texture mip selection and UI still differ from the original.
