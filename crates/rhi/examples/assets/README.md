# Example assets

These files are the exact resources used by the selected ports from
[SaschaWillems/Vulkan](https://github.com/SaschaWillems/Vulkan).  They were
copied without conversion from the `assets` submodule at commit
[`a27c0e584434d59b7c7a714e9180eefca6f0ec4b`](https://github.com/SaschaWillems/Vulkan-Assets/tree/a27c0e584434d59b7c7a714e9180eefca6f0ec4b).

The glTF files in `models/` use embedded buffers and images; no sidecar
resources are required.

## Fetching exact local bytes

Run the reproducible Windows fetcher from `crates/rhi/examples` when the
binary assets are not present:

```powershell
.\fetch-assets.ps1
```

It downloads only the 20 resources listed below from the pinned commit and
checks each SHA-256 digest before placing it in this directory. Existing files
must already match their pinned digest; the script never replaces a mismatched
file. Use `-Destination <path>` to populate a separate local asset directory.

The fetched binaries are deliberately excluded from version control. The
upstream asset pack does not identify a redistributable license for each file;
its Vulkan scene models and derived work specifically require permission for
use or distribution. `fetch-assets.ps1` obtains the exact local bytes needed
to build the examples, but it does not grant a license to redistribute them.

| Example | Resources |
| --- | --- |
| `02_texture` | `textures/metalplate01_rgba.ktx` |
| `03_instancing` | `models/rock01.gltf`, `models/lavaplanet.gltf`, `textures/lavaplanet_rgba.ktx`, `textures/texturearray_rocks_rgba.ktx` |
| `05_push_constants` | `models/sphere.gltf` |
| `06_compute` | `textures/vulkan_11_rgba.ktx` |
| `07_offscreen` | `models/plane.gltf`, `models/chinesedragon.gltf` |
| `08_msaa` | `models/voyager.gltf` |
| `09_texture_mipmap` | `models/tunnel_cylinder.gltf`, `textures/metalplate_nomips_rgba.ktx` |
| `10_indirect_draw` | `models/plants.gltf`, `models/plane_circle.gltf`, `models/sphere.gltf`, `textures/texturearray_plants_rgba.ktx`, `textures/ground_dry_rgba.ktx` |
| `11_occlusion_query` | `models/plane_z.gltf`, `models/teapot.gltf`, `models/sphere.gltf` |
| `12_screenshot` | `models/chinesedragon.gltf` |
| `13_multithreaded_recording` | `models/retroufo_red_lowpoly.gltf`, `models/sphere.gltf` |
| `15_multiview` | `models/sampleroom.gltf` |

`04_dynamic_uniform_buffer` and `14_descriptor_indexing` create their
geometry and texture data at runtime, as do the remaining selected examples
that do not appear in the table.

## Attribution and licenses

The original asset pack's acknowledgements apply to these local copies. It
identifies some glTF sample models as originating from
[KhronosGroup/glTF-Sample-Models](https://github.com/KhronosGroup/glTF-Sample-Models),
credits the Retro UFO model to Sascha Willems under
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/), and states that
Vulkan scene models and derived work are by Dominic Agoro-Ombaka and Sascha
Willems and are not to be used or distributed without request. The pack does
not map the other selected filenames to individual licenses, so all 20
binaries stay local until their redistribution terms are established.

Source: [`Vulkan-Assets` README](https://github.com/SaschaWillems/Vulkan-Assets/blob/a27c0e584434d59b7c7a714e9180eefca6f0ec4b/README.md).
