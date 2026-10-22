# Inline ray query: particles and reflections

A controlled visual subset of NVIDIA Donut Samples `rt_particles`,
revision `d766a7697b67b787714c3f8837fb271583306ab7` (MIT). It builds a BLAS
and TLAS from the five original meshes and traces reflected rays using inline
ray query. It reads back three 1000 x 709 images at 8.363, 10.212 and 12.228
seconds. The reference assets and NVIDIA license are in
`../assets/19_ray_query_particles/`.

On Windows with DXR 1.1 and shader model 6.5:

```text
cargo run -p fluxel-rhi --example 19_ray_query_particles --features examples
```

The program writes `ray-query-particles-01.png` through `-03.png` to the
current directory. Fixed-sequence mode uses deterministic 1/60 second steps;
the original application otherwise uses actual frame durations.
`compare.py REFERENCE ACTUAL` requires Pillow and checks color error outside
the reference UI and motion in the smoke
region. The reference captures must have the same dimensions and fixed camera.

The sample uses the original scene and environment map, smooth mesh normals,
particle simulation, smoke and logo textures, selectable MLAB fragment counts,
and linear-to-sRGB output conversion. This is a visual subset, not an exact
port: primary smoke uses a dynamically rebuilt triangle BLAS and inline
non-opaque triangle candidates. Secondary smoke uses a procedural AABB BLAS,
per-particle TLAS instances, and inline ray-query candidates. Smoke uses a
generated linear-space mip chain and ray-footprint selection.
On the tested AMD GPU, the unmodified NVIDIA sample's AABB stride 0 produced
no visible secondary particle candidates. With only that reference stride set
to 24 bytes, the three RHI frames have mean RGB errors of 0.877, 0.940 and
0.846 outside the UI; they are still not pixel-identical. See the
[source-bound comparison](../../../../documents/evidence/rhi-advanced-0.17/demo19-reference-stride/README.md).
The live window has an interactive settings panel driven by host mouse and
keyboard events. It supports pause, emitter dragging, particle texture,
orientation, ray reorientation and MLAB choices, but uses a simplified bitmap
panel instead of matching the original ImGui appearance and interactions.
Camera controls and some source behaviors remain missing. The fixed-step
readback mode leaves the panel disabled for scene comparison; the live input
captures and exact source hashes are in the
[interactive evidence](../../../../documents/evidence/rhi-advanced-0.17/demo19-interactive/report.md).
DX12 and Vulkan use adapted HLSL shaders. Set
`FLUXEL_RAY_QUERY_PRESENT=1` to run an animation in a window using actual
frame durations and reuse scene resources across frames. As in the source,
emission depends on actual frame timing. For fixed-step comparisons set
`FLUXEL_RAY_QUERY_FIXED_REFERENCE=1` to render frames 120, 240 and 360. Set
`FLUXEL_RAY_QUERY_STATE_DIR` to an existing directory to save the active particle
state alongside those readbacks. Vulkan ray query requires a device
supporting the corresponding Vulkan extensions and features. The NVIDIA
reference executable and captures are external validation inputs.
