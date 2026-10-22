# Example 19: native particle AABB candidates

On Windows MSVC, AMD Radeon 780M, driver 32.0.21028.2002, the example was
built from the source and shader hashes in `sha256.json`. For each backend
run with `FLUXEL_RAY_QUERY_FIXED_REFERENCE=1` and
`FLUXEL_RAY_QUERY_STATE_DIR` set to its output directory. DX12 used
`FLUXEL_DX12_DEBUG=1`; Vulkan used `VK_LAYER_KHRONOS_validation`. Both
produced frames 120, 240 and 360 with empty stderr and exit code zero.
Both backends also ran a live window for five seconds with tracing enabled;
the processes remained healthy and were then terminated by the harness.
The frame 360 DX12 capture was opened and checked.

A shared unit AABB BLAS and per-particle mask-4 TLAS instances feed the
secondary inline ray query, which caps TMax at the opaque hit and applies
the source quaternion orientation and four-layer blend. Primary smoke is
still a software plane approximation. Compared against the captured NVIDIA
reference in `../demo19-fixed-sequence/`, the mean RGB absolute error in
(400, 0, 1000, 709) is 5.351, 5.356 and 6.014 on DX12 (Vulkan agrees to
within 0.00002), versus 4.220, 4.029 and 4.845 in the previous software-only
candidate. Native procedural traversal is more faithful to the source
semantics but the reference's visible reflected smoke remains unexplained.
