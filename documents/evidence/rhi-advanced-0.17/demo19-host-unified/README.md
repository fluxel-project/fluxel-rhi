# Example 19: unified host dependency replay

Source commit: `591ef9188b7b4368bf37758b4d7485f8d51b3c45`. The RHI, DX12 and GL
manifests and `Cargo.lock` all resolve `fluxel-host` to
`b310f3fac4b8319343ef13341cd284344343c2ac`; the host `origin/main` was
verified at that revision. `sha256.json` binds the committed source, lockfile,
executable, logs and referenced screenshots to this replay.

Windows MSVC, AMD Radeon 780M, driver 32.0.21028.2002, 1000 x 709 output.
The executable was rebuilt with Rust 1.87 after unifying the host revision.
`FLUXEL_RAY_QUERY_FIXED_REFERENCE=1` selected deterministic frames 120, 240
and 360. DX12 ran with `FLUXEL_DX12_DEBUG=1`; Vulkan ran with
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`. Each process exited 0 and
printed three expected frame records with no diagnostics in the retained logs.

All six PNG files were byte-for-byte identical to the files in
[`demo19-interactive`](../demo19-interactive/report.md), so this packet refers
to those retained images and records their hashes instead of duplicating
them. The previous packet contains live input captures, which already used
the new host runtime; this replay specifically checks the unified dependency
and fixed-frame GPU path. It does not establish pixel identity with NVIDIA's
reference or complete its ImGui and camera interactions.
