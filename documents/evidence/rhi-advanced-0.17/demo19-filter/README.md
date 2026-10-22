# Example 19: bilinear texture sampling

The environment and smoke textures now use wrapping bilinear mip-zero sampling
instead of nearest texels. Smoke color is decoded from sRGB before interpolation,
while alpha remains linear. The original uses a wrapping sampler, mip zero for
the environment and ray-footprint based mip levels for particles. This change
does not implement particle mip selection or anisotropic filtering.

The NVIDIA reference is Donut Samples `rt_particles` revision
`d766a7697b67b787714c3f8837fb271583306ab7`. Captures were made on Windows
MSVC with AMD Radeon 780M and driver 32.0.21028.2002 from a worktree based on
`0917271` with the example shader changes applied. `sha256.json` records the
source, compiled shaders, executable and evidence hashes. Build and run with:

```text
cargo build -p fluxel-rhi --features examples,dx12,vulkan --example 19_ray_query_particles
19_ray_query_particles.exe --backend dx12
19_ray_query_particles.exe --backend vulkan
```

Each run used a separate working directory; Vulkan enabled
`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`. Both exited successfully and
produced three 1000 x 709 GPU readbacks. The existing `compare.py` passed on
both backends; its output and the command logs are retained. The third DX12
frame was opened and shows the reflective geometry and colored smoke.

`comparison.json` compares the original images at `../visual/19/reference-*.png`
against the preceding MLAB-only output at `../demo19-mlab/dx12-*.png` and the
new output. The RGB mean absolute error in region (400, 0, 1000, 709) changed
from 6.812 to 6.790, 6.538 to 6.520 and 6.758 to 6.735. The DX12/Vulkan mean
difference in the same region was below 0.0004 for each frame. Comparison is
not proof of an exact port: the frames use different simulation time steps and
the remaining geometry, reflection, mip and interaction differences are listed
in the example README.
