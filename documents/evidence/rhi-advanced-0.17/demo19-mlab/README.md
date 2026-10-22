# Example 19: four-fragment blending

These captures exercise the default four-fragment depth-sorted MLAB blending
from NVIDIA Donut Samples `rt_particles` revision
`d766a7697b67b787714c3f8837fb271583306ab7`. The previous example blended in
particle storage order. The shader and executable hashes are in `sha256.json`;
this capture was made from a worktree based on commit `0917271` with the
example shader changes applied, before a source commit was assigned.

On Windows MSVC with AMD Radeon 780M (driver 32.0.21028.2002), build with
`cargo build -p fluxel-rhi --features examples,dx12,vulkan --example 19_ray_query_particles`.
Run the resulting executable with `--backend dx12` and `--backend vulkan` in
separate directories. Vulkan used `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`.
Both commands completed successfully, producing three 1000 x 709 readbacks;
`dx12.log` and `vulkan.log` retain the outputs. The first DX12 frame and the
original reference frame were opened for visual inspection. Both contain the
scene, reflected spheres and colored smoke, with differences in the reflected
smoke and reference UI.

`comparison.json` records RGB mean absolute errors in the visible region
(400, 0, 1000, 709) and smoke region (400, 300, 650, 650). The original
reference frames are `../visual/19/reference-*.png`; the baseline frames are
`../frozen-cbb7070/readback/dx12-19-ray-query-*.png`. Against the original,
visible-region errors for the old and new results were 6.867/6.812,
6.555/6.538 and 6.588/6.758 respectively. DX12 and Vulkan differ by less
than 0.0005 mean RGB levels in that region for each frame. This is evidence
for the blending change on this device; it does not establish an exact port.
The reference uses actual frame timing, geometry and AABB particle intersections,
texture filtering, and additional orientation modes. The example still uses
fixed simulation steps, camera-facing planes, nearest texel sampling and three
precomputed presentation states.
