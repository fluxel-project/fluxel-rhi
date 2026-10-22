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
current directory. The animation uses deterministic 1/60 second steps; the
reference uses actual frame durations. `compare.py REFERENCE ACTUAL` requires
Pillow and checks color error outside the reference UI and motion in the smoke
region. The reference captures must have the same dimensions and fixed camera.

The sample uses the original scene and environment map, smooth mesh normals,
particle simulation and smoke texture, and linear-to-sRGB output conversion.
This is a visual subset, not an exact port: it renders offline, samples textures
at the nearest texel, blends particles in storage order instead of the source's
bounded depth-sorted array, and approximates both primary and reflected smoke
with camera-facing planes instead of the source's triangle and procedural-AABB
intersection paths. It does not implement the UI or all original material and
particle orientation modes. DX12 uses the original HLSL and Vulkan uses an ABI-adapted SPIR-V shader. Set `FLUXEL_RAY_QUERY_PRESENT=1` to present the three precomputed GPU frames in a window, switching every two seconds. This presentation is a controlled sequence, not a continuously simulated animation. Vulkan ray query requires a device supporting the corresponding Vulkan extensions and features. The
NVIDIA reference executable and captures are external validation inputs.
