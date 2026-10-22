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
particle simulation and smoke texture, the default four-fragment MLAB blending,
and linear-to-sRGB output conversion. This is a visual subset, not an exact
port: primary smoke uses a dynamically rebuilt triangle BLAS and inline
non-opaque triangle candidates. Secondary smoke uses a procedural AABB BLAS,
per-particle TLAS instances, and inline ray-query candidates. Smoke uses a
generated linear-space mip chain and ray-footprint selection.
Secondary reflections remain visually different from the source. It does not
implement the UI or all original material and particle orientation modes. DX12
and Vulkan use adapted HLSL shaders. Set
`FLUXEL_RAY_QUERY_PRESENT=1` to run an animation in a window using actual
frame durations and reuse scene resources across frames. As in the source,
emission depends on actual frame timing. For fixed-step comparisons set
`FLUXEL_RAY_QUERY_FIXED_REFERENCE=1` to render frames 120, 240 and 360. Set
`FLUXEL_RAY_QUERY_STATE_DIR` to an existing directory to save the active particle
state alongside those readbacks. Vulkan ray query requires a device
supporting the corresponding Vulkan extensions and features. The NVIDIA
reference executable and captures are external validation inputs.
