# RHI conformance matrix

**Scope:** current checkout. The facade exposes DX12 and Vulkan on Windows and WebGPU in browsers. GL and Metal are not active workspace backends.

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

The DX12 column identifies implemented example paths, not a fresh hardware pass for every row. Vulkan and WebGPU screenshot comparisons use the local C++ reference captures, excluding the C++ HUD. The last 15-example Vulkan regression ran each example for two frames with `VK_LAYER_KHRONOS_validation`; every process exited successfully with empty stderr. The 13–15 DX12 debug-layer paths were also rerun after their latest changes with no diagnostics. Earlier DX12 examples have their own evidence and must be rerun after relevant backend changes. Browser fallback scenes prove visible output only; they do not prove the unavailable native capabilities in their row.

## Updating this matrix

Add a row only after recording a reproducible command and result. Keep source compilation, validation-layer execution, and visual comparison separate. Do not infer unexercised backend capabilities from these examples.
