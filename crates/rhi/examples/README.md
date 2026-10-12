# RHI examples

The 15 numbered examples use the statically typed `fluxel_rhi::api` and the
shared `fluxel-host` window lifecycle. DX12 is the established Windows route.
Vulkan support is being migrated one example at a time. Examples 01–12 have
been run for two frames on Vulkan with the Khronos validation layer and
GPU-image readback comparison against the local C++ captures. Later numbered
examples still use DX12.

From the workspace root on Windows:

```powershell
.\crates\rhi\examples\fetch-assets.ps1
cargo run -p fluxel-rhi --features examples,dx12 --example 01_triangle -- --frames 2
```

Run one of examples 01–12 with the Vulkan feature and explicit backend selection:

```powershell
cargo run -p fluxel-rhi --features examples,vulkan --example 01_triangle -- --backend vulkan --frames 2
```

The native host accepts `--backend dx12|vulkan` and `--frames COUNT` for
examples 01–12. Backend selection happens once at startup; GPU objects and
command recording retain their concrete backend type. The example host and
submission helper live in `common/`. The GPU workload remains in each
numbered example or its small sibling modules.

Examples 01–12 use checked-in DXIL for DX12 and SPIR-V for Vulkan. The SPIR-V
blobs are built from adjacent HLSL sources with `compile-spirv.ps1`; the
script also validates the result for Vulkan 1.3. `fetch-assets.ps1` obtains
the pinned external models and textures required by some examples. Those
assets remain local; see [assets/README.md](assets/README.md).

| Example | Workload |
| --- | --- |
| 01 triangle | Indexed triangle and presentation |
| 02 texture | Textured, lit quad with uniform and sampler bindings |
| 03 instancing | glTF rock field, planet, stars, depth, and instanced draw |
| 04 dynamic uniform buffer | Cube grid with per-draw dynamic uniform offsets |
| 05 push constants | Push constants |
| 06 compute | Compute dispatch |
| 07 offscreen | Offscreen pass sampled by a display pass |
| 08 msaa | Multisampling and resolve |
| 09 texture mipmap | Mipmap generation and sampling |
| 10 indirect draw | Native indirect drawing |
| 11 occlusion query | GPU occlusion query and readback |
| 12 screenshot | Texture readback |
| 13 multithreaded recording | Secondary/native parallel recording |
| 14 descriptor indexing | Capability-gated runtime descriptor indexing |
| 15 multiview | Capability-gated multiview rasterization |

The visual reference is the local `SaschaWillems/Vulkan` C++ checkout. Compare
actual frames with the reference captures under `临时/参考效果/`; the C++ HUD in
the upper-left corner is outside the rendering comparison. Follow the public
coordinate contract in `documents/design-rhi.md` when interpreting a vertical
flip or front-face difference. A backend must adapt its native viewport and
winding rather than changing the example's shared scene data.
