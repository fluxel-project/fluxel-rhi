# Example 19: native primary particle triangles

On Windows MSVC, AMD Radeon 780M, driver 32.0.21028.2002, the example was
built from the source and shader hashes in `sha256.json`. For each backend
run with `FLUXEL_RAY_QUERY_FIXED_REFERENCE=1` and
`FLUXEL_RAY_QUERY_STATE_DIR` set to its output directory. DX12 used
`FLUXEL_DX12_DEBUG=1`; Vulkan used `VK_LAYER_KHRONOS_validation`. Both
produced frames 120, 240 and 360 with empty stderr and exit code zero.
Both backends also ran a live window for five seconds with tracing enabled;
the processes remained healthy and were then terminated by the harness.
The frame 360 DX12 capture was opened and checked.

The primary ray queries a non-opaque particle triangle BLAS with mask 2.
Its two primitives per active particle use the same compact ordering as the
particle data. The TLAS also contains opaque geometry with mask 1 and
procedural particle AABBs with mask 4. The per-frame triangle BLAS and TLAS
are rebuilt after previous GPU completion; resources are reused.

Compared against the captured NVIDIA reference in `../demo19-fixed-sequence/`,
the mean RGB absolute error in (400, 0, 1000, 709) is 5.302, 5.269 and
5.832 on DX12 (Vulkan agrees within 0.00003). The preceding native AABB
candidate measured 5.351, 5.356 and 6.014. This remains worse than the older
software-only candidate (4.220, 4.029 and 4.845): reflected smoke differs
visibly from the original for reasons still under investigation.
