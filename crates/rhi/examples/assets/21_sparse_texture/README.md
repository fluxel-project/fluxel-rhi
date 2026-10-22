# Microsoft ReservedResources shader

`shaders.hlsl` originates from Microsoft DirectX-Graphics-Samples revision `e5975f9b0744fc5096dd593ed72bf8f5c004164f`; the included MIT license applies. The RHI shader changes the vertex input semantics to LOCATION0 and LOCATION1, maps the texture and sampler to bindings 1 and 2, and samples mip zero of a view restricted to the selected underlying mip. DXIL was built with DXC 1.9.0.5399 using `-T vs_6_0 -E VSMain` and `-T ps_6_0 -E PSMain`.

This example renders the official checkerboard across all nine mips of a 256x256 RGBA8 sparse texture. Standard mips have separate allocations; packed mips share their reported tail allocation. The selected region is mapped and the previous region unmapped; selection cycles automatically or can be fixed with `FLUXEL_SPARSE_MIP` (0 through 8). The official sample switches with keyboard input. The original strip is expressed as two triangles for the current DX12 pipeline.

SHA-256:

- `shaders.hlsl`: `0957f7e5b7a5373e53d055b280ab44dbe796643efb994dd5b4a2dca7e33d8c9b`
- `shaders.rhi.hlsl`: `a3f426a73fcdf926b94ae31faf0a591d94537fbb02ad4f6ce88181b9ed6b9ea3`
- `vs.dxil`: `0405248cb4b6a2419b518987e91430c5b5adce4cbfe14d032e9087be2c15de70`
- `ps.dxil`: `cd390a7d5d8b48be47c9316b93679394c20491fd84c1ddf8b5cef0db20229560`
