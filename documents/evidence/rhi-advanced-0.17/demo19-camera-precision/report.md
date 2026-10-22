# Example 19: camera precision

Built on Windows MSVC from worktree base `8ced4924e60238ef6e50f4bc42c976bb2273ba43`
with the example changes listed by `sha256.json`, using an AMD Radeon 780M
with driver 32.0.21028.2002. Both runs set `FLUXEL_RAY_QUERY_FIXED_REFERENCE=1`
and `FLUXEL_RAY_QUERY_STATE_DIR`; DX12 used `FLUXEL_DX12_DEBUG=1`, Vulkan
used `VK_LAYER_KHRONOS_validation`. Both returned zero with empty stderr.
Images and particle state cover frames 120, 240 and 360, at 1000 x 709.

The camera constants now retain precision derived from the source camera
target, 6-unit distance and yaw/pitch angles, including the particle quad axes.
The ray cone angle also uses the actual output height rather than a 709
literal; the current window itself remains fixed at 709 pixels high.

Comparing the UI-free rectangle (400, 0, 1000, 709) to the NVIDIA reference
with stride 24 in `../demo19-reference-stride/`, DX12 mean absolute RGB
errors for frames 120/240/360 were 0.877/0.940/0.846; previously they were
0.885/0.949/0.855. Vulkan agrees to within 0.00003. The reference and DX12
frame 360 were visually inspected. The result is closer but remains a visual
subset rather than a 100% port.
