# Raytracing Hello World source

`Raytracing.hlsl` and `RaytracingHlslCompat.h` are copied unchanged from
[Microsoft's DirectX Graphics Samples](https://github.com/microsoft/DirectX-Graphics-Samples/tree/e5975f9/Samples/Desktop/D3D12Raytracing/src/D3D12RaytracingHelloWorld),
commit `e5975f9`. They are distributed under the included Microsoft MIT license.
The source project compiles the library with shader model 6.3. The checked-in
DXIL was generated using:

```powershell
dxc -T lib_6_3 -Fo Raytracing.dxil Raytracing.hlsl
dxc -T lib_6_3 -Fo Raytracing_rhi.dxil Raytracing_rhi.hlsl
dxc -T vs_6_0 -E vs_main -Fo display_vs.dxil display.hlsl
dxc -T ps_6_0 -E ps_main -Fo display_ps.dxil display.hlsl
```

`Raytracing_rhi.hlsl` changes only the output UAV register from `u0` to `u1`:
the RHI binding numbers are unique within a group and map to DX12 register
numbers. The scene and shading calculations are unchanged. SHA-256 of the
original is `88f43b9a61cf6bdb4512794b618d857b8805b15b39dfbdba85352cb5e7484f76`;
the adapted file is `2a71e1d57e648d378070077ea4cb5d77f814e1abd0a0a17d9d41323c04bcce0c`.

The official sample uses a 1280 by 720 window, one indexed triangle and a
single identity-transformed instance. The shader writes a barycentric triangle
inside a black stencil, with a red/green dispatch-coordinate gradient outside.
The stencil is adjusted to preserve a fixed border as the window changes size.

On Windows, set `FLUXEL_RAY_READBACK` to a file path to save the second frame's
GPU output as RGB PPM. The first submission builds the scene, then the example
waits for completion and drops its BLAS and build descriptions before tracing
again. This exercises the TLAS ownership of its referenced BLAS. The display
pass uses integer texel loads. Compare captures with
`tools/examples/compare-ray-hello.py`.

For Vulkan, `Raytracing_vulkan.hlsl` adapts the RHI shader's local constant
buffer to `vk::shader_record_ext` and declares the output image as `rgba8`.
The scene and shading remain the same. Run
`python tools/examples/build-native-spirv.py --verify` from the repository root
to rebuild and validate all checked-in native SPIR-V artifacts.
