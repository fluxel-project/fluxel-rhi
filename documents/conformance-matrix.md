# RHI conformance matrix

**Scope:** current checkout. The facade exposes native DX12 and Vulkan backends on Windows. GL, Metal, and WebGPU are not active workspace backends.

`Pass` means a recorded native run, not successful compilation. `Unsupported` requires an explicit capability or operation result. A successful example proves its exercised path only, not every operation in the backend API.

| Route | DX12 example path | Vulkan evidence |
| --- | --- | --- |
| Typed Core/facade integration | Workspace check and all examples build | Workspace check and all examples build |
| Surface acquire, graphics submit, present | Numbered Windows examples | Examples 01–15, two frames each with validation layer |
| Texture upload, views, samplers, depth | Examples 02–09 | Examples 02–09 |
| Compute dispatch | Example 06 | Example 06 |
| Texture array, indirect and multi-draw | Example 10 | Example 10 |
| Occlusion query and readback | Example 11 | Example 11; observed teapot 246604 and sphere 2878 samples |
| Image readback | Example 12 | Example 12; PPM and GPU capture pixel-identical |
| Secondary raster work | Example 13 | Example 13; native secondary command buffers |
| Runtime descriptor indexing | Example 14 | Example 14; variable descriptor count and nonuniform image indexing |
| Two-view multiview | Example 15 | Example 15; native view mask `0b11` and `ViewIndex` |

The DX12 column identifies implemented example paths, not a fresh hardware pass for every row. Vulkan screenshot comparisons use the local C++ reference captures, excluding the C++ HUD. The last 15-example Vulkan regression ran each example for two frames with `VK_LAYER_KHRONOS_validation`; every process exited successfully with empty stderr. The 13–15 DX12 debug-layer paths were also rerun after their latest changes with no diagnostics. Earlier DX12 examples have their own evidence and must be rerun after relevant backend changes.

## Updating this matrix

Add a row only after recording a reproducible command and result. Keep source compilation, validation-layer execution, and visual comparison separate. Do not infer unexercised backend capabilities from these examples.
