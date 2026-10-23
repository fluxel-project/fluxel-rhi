# RHI conformance matrix


The optional common execution entry is verified separately from native feature
capabilities. Migration checks covered source fingerprints, hardware readbacks
and platform limits. They do not add GL deferred recording, native multi-queue
support, or Metal GPU acceptance.

**Scope:** the DX12, Vulkan, and WebGPU columns summarize examples 01–15.
DX12/Vulkan examples 16–23 also received focused GPU fixtures and reference
comparisons; that candidate's source was not frozen during all captures. GL4+,
GLES3+, and WebGL2 have separate profile evidence in
`crates/rhi/examples/README.md`. Metal has Apple-target compile checks and
static review, but no Apple GPU evidence, so it is intentionally not represented
as a passing matrix column.

`Pass` means a recorded native run, not successful compilation. `Unsupported` requires an explicit capability or operation result. A successful example proves its exercised path only, not every operation in the backend API.

| Route | DX12 example path | Vulkan evidence | WebGPU browser evidence |
| --- | --- | --- | --- |
| Typed Core/facade integration | Workspace check and all examples build | Workspace check and all examples build | WASM check and browser backend tests |
| Surface acquire, graphics submit, present | Numbered Windows examples | Examples 01–15, two frames each with validation layer | Numbered three-frame browser tests |
| Texture upload, views, samplers, depth | Examples 02–09 | Examples 02–09 | Examples 02–09 |
| Compute dispatch | Example 06 | Example 06 | Example 06, native compute pass |
| Texture array and indirect draw | Example 10 | Example 10, including multi-draw | Example 10, texture array and `drawIndexedIndirect`; multi-draw is not advertised |
| Occlusion query and readback | Example 11 | Example 11; observed teapot 246604 and sphere 2878 samples | Example 11, native query resolve and staging readback |
| Image readback | Example 12 | Example 12; PPM and GPU capture pixel-identical | Example 12, texture-to-buffer copy and async map |
| Secondary raster work | Example 13 | Example 13; native secondary command buffers | Unsupported; example 13 preserves the scene using sequential primary recording |
| Runtime descriptor indexing | Example 14 | Example 14; variable descriptor count and nonuniform image indexing | Unsupported; example 14 uses a fixed 32-layer texture for the same scene |
| Two-view multiview | Example 15 | Example 15; native view mask `0b11` and `ViewIndex` | Unsupported; example 15 renders each eye in a separate pass |

The DX12 column identifies implemented example paths, not a fresh hardware pass for every row. Vulkan and WebGPU screenshot comparisons use the local C++ reference captures, excluding the C++ HUD. The current Vulkan candidate ran all 15 examples for two frames on AMD Radeon 780M (driver `25.10.28.02`) with `VK_LAYER_KHRONOS_validation`; every example exited successfully with zero validation diagnostics. Visual review found matching scenes across all 15. Example 09's requested anisotropic sampler is now enabled when supported by the device. Example 14 declares the texture upload, shader-read, and return-to-common transitions explicitly. Example 13's presentation was visually checked from swapchain GPU readback because desktop capture was occluded. Example 07 retains the source demo's intentional 7×7 reflection blur. The 14 shared-example transition change was rerun on DX12 with the debug layer and no diagnostics. WebGPU ran all 15 browser tests for three frames each in Chrome 155.0.8059.40 on an AMD RDNA 3 adapter (`isFallbackAdapter = false`) and produced fresh 1280×720 captures for visual comparison. Its 05/13/14/15 fallback scenes prove visible output only; they do not prove the unavailable native capabilities in their row.

## Updating this matrix

Add a row only after recording a reproducible command and result. Keep source compilation, validation-layer execution, and visual comparison separate. Do not infer unexercised backend capabilities from these examples.
